//! Serves the synthetic engine over HTTP, so the whole path from socket to GPU
//! can be load-tested, and with `--load` does that itself.
//!
//! ```sh
//! # serve until interrupted
//! cargo run --release -p ares-runtime --example runtime_serve -- --port 18900
//! # start the server on a free port, drive it with 32 streaming clients, print the numbers
//! cargo run --release -p ares-runtime --example runtime_serve -- --port 18900 --load 32
//! ```
//!
//! It listens on 127.0.0.1 only, and refuses the ports of other services on
//! this machine (8080, 5803). The clients are raw sockets, one connection per
//! request, speaking `POST /v1/completions` with `"stream": true`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ares_api::Engine;
use ares_cuda::{Device, Wait, devices};
use ares_runtime::testing::HostExecutor;
use ares_runtime::{GpuConfig, Runtime, RuntimeConfig, Split, SyntheticConfig, SyntheticModel};
use ares_server::{Config, serve};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

const HELP: &str = "\
runtime_serve: the synthetic engine behind the HTTP server

OPTIONS
  --port N              port on 127.0.0.1 (required; 0 picks a free one)
  --placement NAME      single, tensor or layer (default single)
  --layers N            layers (default 64)
  --fake                no GPU: a host executor with a fixed step time (--step-us)
  --step-us N           step time of the host executor (default 20000)
  --load N              run N concurrent streaming clients against the server, report, exit
  --rounds R            requests per client with --load (default 3)
  --tokens N            tokens per request with --load (default 128)
  --prompt N            prompt characters with --load (default 24)
  --speculation on|off  n-gram drafts (default off)
";

struct Options {
    port: u16,
    placement: String,
    layers: u32,
    fake: bool,
    step_us: u64,
    load: usize,
    rounds: usize,
    tokens: u32,
    prompt: usize,
    speculation: bool,
}

fn parse() -> Result<Options, String> {
    let mut o = Options {
        port: u16::MAX,
        placement: "single".into(),
        layers: 64,
        fake: false,
        step_us: 20_000,
        load: 0,
        rounds: 3,
        tokens: 128,
        prompt: 24,
        speculation: false,
    };
    let mut port = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        let number = |text: String, name: &str| {
            text.parse::<u64>().map_err(|_| format!("{name}: '{text}' is not a number"))
        };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--port" => port = Some(value(&flag)?.parse::<u16>().map_err(|_| "bad port")?),
            "--placement" => o.placement = value(&flag)?,
            "--layers" => o.layers = number(value(&flag)?, &flag)? as u32,
            "--fake" => o.fake = true,
            "--step-us" => o.step_us = number(value(&flag)?, &flag)?,
            "--load" => o.load = number(value(&flag)?, &flag)? as usize,
            "--rounds" => o.rounds = number(value(&flag)?, &flag)? as usize,
            "--tokens" => o.tokens = number(value(&flag)?, &flag)? as u32,
            "--prompt" => o.prompt = number(value(&flag)?, &flag)? as usize,
            "--speculation" => o.speculation = value(&flag)? == "on",
            other => return Err(format!("unknown option '{other}' (see --help)")),
        }
    }
    let port = port.ok_or("--port is required")?;
    if port == 8080 || port == 5803 {
        return Err(format!("port {port} belongs to another service on this machine"));
    }
    o.port = port;
    Ok(o)
}

/// What one request saw.
struct Outcome {
    first_text: Duration,
    total: Duration,
    tokens: u64,
    ok: bool,
}

/// One streaming completion over a fresh connection.
async fn one_request(addr: std::net::SocketAddr, prompt: &str, tokens: u32) -> Outcome {
    let body = format!(
        "{{\"model\":\"synthetic\",\"prompt\":\"{prompt}\",\"max_tokens\":{tokens},\"stream\":true,\
         \"stream_options\":{{\"include_usage\":true}},\"temperature\":0}}"
    );
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let started = Instant::now();
    let mut outcome =
        Outcome { first_text: Duration::ZERO, total: Duration::ZERO, tokens: 0, ok: false };
    let Ok(mut stream) = TcpStream::connect(addr).await else { return outcome };
    let _ = stream.set_nodelay(true);
    if stream.write_all(request.as_bytes()).await.is_err() {
        return outcome;
    }
    let mut received = Vec::with_capacity(8192);
    let mut chunk = [0u8; 8192];
    let mut seen_text = false;
    loop {
        match stream.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                received.extend_from_slice(&chunk[..n]);
                if !seen_text {
                    // The first chunk that carries text, not the role or an empty piece.
                    let text = String::from_utf8_lossy(&received);
                    if let Some(at) = text.find("\"text\":\"") {
                        let after = &text[at + 8..];
                        if after.chars().next().is_some_and(|c| c != '"') {
                            seen_text = true;
                            outcome.first_text = started.elapsed();
                        }
                    }
                }
            }
        }
    }
    outcome.total = started.elapsed();
    let text = String::from_utf8_lossy(&received);
    outcome.ok = text.starts_with("HTTP/1.1 200") && text.contains("[DONE]");
    if let Some(at) = text.rfind("\"completion_tokens\":") {
        let digits: String = text[at + 20..].chars().take_while(char::is_ascii_digit).collect();
        outcome.tokens = digits.parse().unwrap_or(0);
    }
    outcome
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p).round() as usize]
}

#[tokio::main]
async fn main() {
    let o = match parse() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let (split, used) = match o.placement.as_str() {
        "single" => (Split::Single, 1),
        "tensor" => (Split::Tensor, 2),
        "layer" => (Split::Layer, 2),
        other => {
            eprintln!("unknown placement '{other}'");
            std::process::exit(2);
        }
    };
    let mut sc = SyntheticConfig::realistic(split, used);
    sc.layers = o.layers;
    if o.fake {
        sc.layers = 1;
    }
    let model = Arc::new(SyntheticModel::new(sc).expect("a valid model"));
    let config =
        RuntimeConfig { speculation: o.speculation, pages: 16384, ..RuntimeConfig::default() };
    let runtime = if o.fake {
        let host = HostExecutor::new(model.clone(), model.config().max_rows, config.max_slots)
            .with_devices(used)
            .with_delay(Duration::from_micros(o.step_us));
        Runtime::start(model.clone(), Box::new(host), config).expect("starts")
    } else {
        let count = devices().expect("driver").len();
        assert!(count >= used, "{} needs {used} devices, found {count}", o.placement);
        let open: Vec<Device> =
            (0..used as u32).map(|i| Device::open(i, Wait::Spin).expect("open")).collect();
        let gpu = GpuConfig::new(&config, &*model);
        Runtime::start_gpu(model.clone(), open, config, gpu).expect("starts")
    };
    let runtime = Arc::new(runtime);
    let engine: Arc<dyn Engine> = runtime.clone();

    let listener = TcpListener::bind(("127.0.0.1", o.port)).await.expect("bind 127.0.0.1");
    let addr = listener.local_addr().expect("address");
    println!("serving the synthetic model ({}, {} layers) on http://{addr}", o.placement, o.layers);
    let (stop, stopped) = oneshot::channel::<()>();
    let server = tokio::spawn(serve(listener, engine, Config::default(), async move {
        let _ = stopped.await;
    }));

    if o.load == 0 {
        tokio::signal::ctrl_c().await.expect("signal handler");
        println!("shutting down");
    } else {
        println!(
            "load: {} clients x {} requests of {} tokens, prompt {} characters",
            o.load, o.rounds, o.tokens, o.prompt
        );
        // One request first, so connections and caches are warm.
        let warm = one_request(addr, "warm up the path", 8).await;
        assert!(warm.ok, "the server did not answer a request");
        runtime.take_snapshot();
        let started = Instant::now();
        let clients: Vec<_> = (0..o.load)
            .map(|c| {
                let (rounds, tokens, plen) = (o.rounds, o.tokens, o.prompt);
                tokio::spawn(async move {
                    let mut all = Vec::new();
                    for r in 0..rounds {
                        let prompt: String = (0..plen)
                            .map(|k| char::from(b'a' + ((c * 7 + r * 5 + k * 3) % 26) as u8))
                            .collect();
                        all.push(one_request(addr, &prompt, tokens).await);
                    }
                    all
                })
            })
            .collect();
        let mut outcomes = Vec::new();
        for client in clients {
            outcomes.extend(client.await.expect("client"));
        }
        let wall = started.elapsed();
        let failed = outcomes.iter().filter(|r| !r.ok).count();
        let tokens: u64 = outcomes.iter().map(|r| r.tokens).sum();
        let mut ttft: Vec<f64> =
            outcomes.iter().filter(|r| r.ok).map(|r| r.first_text.as_secs_f64() * 1e3).collect();
        ttft.sort_by(f64::total_cmp);
        let mut per_request: Vec<f64> = outcomes
            .iter()
            .filter(|r| r.ok)
            .map(|r| r.tokens as f64 / r.total.as_secs_f64())
            .collect();
        per_request.sort_by(f64::total_cmp);
        println!(
            "\n{} requests in {:.2} s ({} failed)",
            outcomes.len(),
            wall.as_secs_f64(),
            failed
        );
        println!("tokens per second, all clients: {:.1}", tokens as f64 / wall.as_secs_f64());
        println!(
            "tokens per second per request:  median {:.1}  p10 {:.1}  p90 {:.1}",
            percentile(&per_request, 0.5),
            percentile(&per_request, 0.1),
            percentile(&per_request, 0.9)
        );
        println!(
            "time to first token (ms):       median {:.1}  p90 {:.1}  p99 {:.1}  max {:.1}",
            percentile(&ttft, 0.5),
            percentile(&ttft, 0.9),
            percentile(&ttft, 0.99),
            ttft[ttft.len() - 1]
        );
        println!("\n{}", runtime.snapshot().expect("snapshot"));
    }

    let _ = stop.send(());
    let _ = tokio::time::timeout(Duration::from_secs(15), server).await;
    runtime.shutdown();
}
