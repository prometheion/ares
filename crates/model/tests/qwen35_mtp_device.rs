//! The MTP block on a device against its CPU reference, on the real 27B.
//!
//! Needs a GPU (`ARES_TEST_CUDA=1`), the model (`ARES_TEST_QWEN38_27B`) and a
//! capture of it made by `tools/capture` (`ARES_TEST_QWEN35_CAPTURE27`, with
//! `german/base` and `english/base`): the tokens of a prompt and of what
//! follows it, and the model's final hidden state of every position. Only the
//! MTP block, the embedding rows it keeps and the output head go to the device
//! (two gigabytes); the hidden states the model would compute are put there by the
//! test, so the block is compared on its own:
//!
//! - it follows the prompt (its keys and values of every position);
//! - a chain of drafts from the prompt's end equals the reference's chain
//!   token for token, and each link's output is close to the reference's;
//! - after a window whose rows are accepted and committed, the next chain is
//!   again the reference's: the keys and values of the window's rows, the
//!   hidden state kept for the next step and the commit are right;
//! - with sampling, a link's pick is the sampler's pick from the link's
//!   logits.
//!
//! `ARES_TEST_DRAFT_VOCAB` names a draft vocabulary (a token id per line, `#`
//! comments), `ARES_TEST_DEVICE` the device (default 0),
//! `ARES_TEST_QWEN35_MTP_CAPTURE` the fork's chain for the same hidden states
//! (printed next to ours). Device time per link is printed.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ares_core::Ledger;
use ares_cuda::{
    Device, DeviceBuffer, ENTRY_CELLS, Kernels, Precision, Segment, StepRecord, Stream, Wait,
};
use ares_gguf::Gguf;
use ares_model::qwen35::device::{
    MtpChain, MtpStep, Options, PAGE_ENTRY, Part, PartSetup, Placement, Profile, StepEnv,
    page_bytes,
};
use ares_model::qwen35::{Config, MtpState, Reference, Weights};
use ares_runtime::{DraftChances, PinnedRam};
use ares_sampler::{SAMPLE_CANDIDATES, SampleParams, SamplePick, SampleRow, sample_reference};
use common::helpers::{env_path, floats, read_f32};

mod common;

fn read_i32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes.as_chunks::<4>().0.iter().map(|b| i32::from_le_bytes(*b) as u32).collect()
}

fn read_ids(path: &Path) -> Vec<u32> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.split_whitespace().map(|t| t.parse().unwrap()).collect()
}

fn read_vocabulary(path: &Path) -> Vec<u32> {
    ares_model::files::read_draft_vocabulary(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (f64::from(*x) - f64::from(*y)).powi(2);
        n += f64::from(*y).powi(2);
    }
    (d / n).sqrt()
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The MTP block on a device, with the buffers the runtime would own.
struct Rig {
    stream: Stream,
    part: Part,
    record: DeviceBuffer,
    tables: DeviceBuffer,
    pages: DeviceBuffer,
    d: usize,
}

impl Rig {
    fn open(
        device: &Device,
        c: &Config,
        w: &Weights<'_>,
        placement: &Placement,
        opts: &Options,
    ) -> Self {
        let cells = 1024u32;
        let table_stride = cells.div_ceil(ENTRY_CELLS);
        let kernels = Kernels::load(device).unwrap();
        let mut ledger =
            Ledger::new(format!("CUDA{}", device.ordinal()), device.mem_info().unwrap().free);
        let pool = u64::from(table_stride) * page_bytes(c, placement, 0, opts);
        let ram = PinnedRam::new(0);
        let part = Part::open(
            c,
            w,
            placement,
            opts,
            PartSetup {
                index: 0,
                device,
                kernels: &kernels,
                ledger: &mut ledger,
                max_slots: 1,
                pages: table_stride,
                table_stride,
                checkpoints: 0,
                ram: &ram,
                table: None,
            },
        )
        .unwrap();
        let mut rig = Self {
            stream: Stream::new(device, 0).unwrap(),
            part,
            record: DeviceBuffer::new(device, 64 * 1024).unwrap(),
            tables: DeviceBuffer::new(device, u64::from(table_stride) * 4).unwrap(),
            pages: DeviceBuffer::new(device, pool).unwrap(),
            d: c.n_embd,
        };
        let entries: Vec<u32> = (0..table_stride).map(|p| p | PAGE_ENTRY).collect();
        let bytes: Vec<u8> = entries.iter().flat_map(|e| e.to_le_bytes()).collect();
        rig.tables.write(0, &bytes).unwrap();
        rig.part.update_table(0, 0, entries.iter().copied()).unwrap();
        rig
    }

    /// Uploads a record of one segment and returns it with the offset of its
    /// token array.
    fn upload(&self, segment: Segment, tokens: &[u32]) -> (StepRecord, u64) {
        let record = StepRecord::new(&[segment], tokens, None, &[]).unwrap();
        let bytes = record.as_bytes();
        self.record.write(0, bytes).unwrap();
        let off_tokens = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        (record, u64::from(off_tokens))
    }

    /// A step of `tokens` from `pos0` whose final hidden states are `hidden`;
    /// `direct` as the segment would be in a step. The MTP block follows the
    /// rows of a direct step at once (as the next step would); a window's
    /// rows wait for [`Rig::commit`].
    fn follow(&mut self, pos0: u32, tokens: &[u32], hidden: &[f32], direct: bool) {
        let segment =
            Segment { row0: 0, n_rows: tokens.len() as u32, slot: 0, pos0: pos0 as i32, direct };
        let (record, _) = self.upload(segment, tokens);
        let segments = [segment];
        let env = StepEnv {
            stream: &self.stream,
            step: self.record.ptr(),
            dims: record.dims(),
            tokens,
            segments: &segments,
            tables: self.tables.ptr(),
            pages: self.pages.ptr(),
            exchange: None,
            replay: &[],
        };
        self.part.enqueue_mtp_step(&env, hidden).unwrap();
        if direct {
            self.part.mtp_follow(&self.stream, self.tables.ptr(), self.pages.ptr()).unwrap();
        }
        self.stream.synchronize().unwrap();
    }

    /// The outcome of the last step's window: `accepted` rows count, and the
    /// MTP block follows them.
    fn commit(&mut self, accepted: u32) {
        self.part.commit_windows([(0, accepted)]);
        self.part.mtp_follow(&self.stream, self.tables.ptr(), self.pages.ptr()).unwrap();
        self.stream.synchronize().unwrap();
    }

    /// Runs a chain of `rows.len()` drafts after `token` at `pos0` and returns
    /// the drafts, each link's output and pick, the logits of the last link,
    /// and the tokens of the step record after the chain.
    fn chain(&mut self, pos0: u32, token: u32, rows: &[SampleRow]) -> Chain {
        self.chain_with(pos0, token, rows, Vec::new(), None)
    }

    /// [`Rig::chain`] with draft bars: `needs` per link, `chances` the chance
    /// of a draft by its probability.
    fn chain_with(
        &mut self,
        pos0: u32,
        token: u32,
        rows: &[SampleRow],
        needs: Vec<f32>,
        chances: Option<DraftChances>,
    ) -> Chain {
        let k = rows.len();
        let segment =
            Segment { row0: 0, n_rows: k as u32 + 1, slot: 0, pos0: pos0 as i32, direct: false };
        let mut tokens = vec![0u32; k + 1];
        tokens[0] = token;
        let (record, off_tokens) = self.upload(segment, &tokens);
        let segments = [segment];
        let env = StepEnv {
            stream: &self.stream,
            step: self.record.ptr(),
            dims: record.dims(),
            tokens: &tokens,
            segments: &segments,
            tables: self.tables.ptr(),
            pages: self.pages.ptr(),
            exchange: None,
            replay: &[],
        };
        self.part
            .mtp_step(MtpStep {
                tokens: self.record.ptr().offset(off_tokens),
                chains: vec![MtpChain { segment: 0, token, rows: rows.to_vec(), needs, place: 0 }],
                chances,
            })
            .unwrap();
        self.part.enqueue_chains(&env).unwrap();
        self.stream.synchronize().unwrap();
        self.part.flush_profile().unwrap();

        let drafted = self.part.drafted().expect("the step drafted");
        assert_eq!(drafted.tokens as usize, k);
        let (scratch, _) = self.part.mtp_scratch(0).unwrap();
        let mut bytes = vec![0u8; k * 4];
        scratch.read(drafted.ptr.as_raw() - scratch.ptr().as_raw(), &mut bytes).unwrap();
        let drafts: Vec<u32> =
            bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        let mut outputs = Vec::new();
        let mut picks = Vec::new();
        let mut logits = Vec::new();
        for link in 0..k as u32 {
            let (scratch, [out, logits_at, picks_at]) = self.part.mtp_scratch(link).unwrap();
            let mut bytes = vec![0u8; self.d * 4];
            scratch.read(out, &mut bytes).unwrap();
            outputs.push(floats(&bytes));
            let mut bytes = vec![0u8; SamplePick::BYTES];
            scratch.read(picks_at, &mut bytes).unwrap();
            picks.push(SamplePick::from_bytes(&bytes));
            if link as usize + 1 == k {
                let mut bytes = vec![0u8; self.part.draft_head_rows() as usize * 4];
                scratch.read(logits_at, &mut bytes).unwrap();
                logits = floats(&bytes);
            }
        }
        let mut bytes = vec![0u8; (k + 1) * 4];
        self.record.read(off_tokens, &mut bytes).unwrap();
        let record_tokens =
            bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        Chain { drafts, outputs, picks, logits, record_tokens }
    }
}

struct Chain {
    drafts: Vec<u32>,
    outputs: Vec<Vec<f32>>,
    picks: Vec<SamplePick>,
    logits: Vec<f32>,
    record_tokens: Vec<u32>,
}

#[test]
fn the_mtp_block_on_the_device_drafts_what_the_reference_drafts() {
    if !common::gpu::enabled() {
        return;
    }
    let Some(model) = env_path("ARES_TEST_QWEN38_27B") else { return };
    let Some(capture) = env_path("ARES_TEST_QWEN35_CAPTURE27") else { return };
    let fork = std::env::var_os("ARES_TEST_QWEN35_MTP_CAPTURE").map(PathBuf::from);
    let vocabulary =
        std::env::var_os("ARES_TEST_DRAFT_VOCAB").map(|p| read_vocabulary(Path::new(&p)));
    let ordinal = std::env::var("ARES_TEST_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);

    let gguf = Gguf::open(&model).unwrap();
    let config = Config::from_gguf(&gguf).unwrap();
    let weights = Weights::load(&gguf, &config).unwrap();
    let reference = Reference::new(&config, &weights);
    let d = config.n_embd;

    // Only the MTP block, the embedding rows it keeps and the head on the device.
    let mut placement = Placement::single(&config);
    placement.devices[0].blocks = vec![None; config.n_layer];
    let placement = placement.with_mtp(&config, vocabulary.as_deref()).unwrap();
    let draft = placement.draft.clone().unwrap();
    // The tokens the draft head can pick, for the reference.
    let draft_tokens: Vec<u32> = (0..draft.head_rows).map(|r| placement.head_token(r)).collect();
    const LINKS: usize = 4;
    const WINDOW: usize = 5;
    // The reference's chains, before a device is opened: per prompt the chain
    // from the prompt's end and the chain after a window of WINDOW rows.
    let mut wanted = Vec::new();
    for prompt_name in ["german", "english"] {
        let base = capture.join(prompt_name).join("base");
        let prompt = read_ids(&base.join("prompt.tokens"));
        let generated = read_ids(&base.join("generated.tokens"));
        let h_prompt = read_f32(&base.join("prefill.result_norm.f32"));
        let h_steps = read_f32(&base.join("steps.result_norm.f32"));
        let n = prompt.len();
        let mut hidden = vec![0.0f32; d];
        hidden.extend_from_slice(&h_prompt[..(n - 1) * d]);
        let mut state = MtpState::new();
        let rows = reference.mtp_rows(&state, &prompt, &hidden, false).unwrap();
        reference.mtp_commit(&mut state, &rows, n).unwrap();
        let first = reference
            .mtp_draft(&state, generated[0], &h_prompt[(n - 1) * d..], LINKS, Some(&draft_tokens))
            .unwrap();
        let mut before = h_prompt[(n - 1) * d..].to_vec();
        before.extend_from_slice(&h_steps[..(WINDOW - 1) * d]);
        let rows = reference.mtp_rows(&state, &generated[..WINDOW], &before, false).unwrap();
        reference.mtp_commit(&mut state, &rows, WINDOW).unwrap();
        let second = reference
            .mtp_draft(
                &state,
                generated[WINDOW],
                &h_steps[(WINDOW - 1) * d..WINDOW * d],
                LINKS,
                Some(&draft_tokens),
            )
            .unwrap();
        wanted.push([first, second]);
    }

    let device = Device::open(ordinal, Wait::Block).unwrap();
    println!(
        "{}: draft head of {} rows{}",
        device.info().name,
        draft.head_rows,
        if vocabulary.is_some() { " (draft vocabulary)" } else { "" }
    );
    for precision in [Precision::Fast, Precision::Exact] {
        let profile = Arc::new(Profile::new());
        let opts = Options {
            precision,
            max_rows: 256,
            logit_rows: 8,
            window_rows: 8,
            profile: Some(profile.clone()),
            ..Options::default()
        };
        let mut rig = Rig::open(&device, &config, &weights, &placement, &opts);
        let bytes = rig.part.mtp_bytes().unwrap();
        println!(
            "{precision:?}: the part holds {:.1} MiB of weights (kept embedding rows, head, MTP block); the \
             MTP block: weights {:.1} MiB, K/V {:.3} MiB per 1000 tokens, kept hidden states \
             {:.2} MiB, chain scratch {:.2} MiB, step records {:.2} MiB, step scratch grown by {:.2} MiB",
            rig.part.weight_bytes() as f64 / 1048576.0,
            bytes.weights as f64 / 1048576.0,
            bytes.page as f64 / 128.0 * 1000.0 / 1048576.0,
            bytes.carry as f64 / 1048576.0,
            bytes.scratch as f64 / 1048576.0,
            bytes.records as f64 / 1048576.0,
            bytes.step_scratch as f64 / 1048576.0,
        );

        for (prompt_name, wanted) in ["german", "english"].into_iter().zip(&wanted) {
            let base = capture.join(prompt_name).join("base");
            let prompt = read_ids(&base.join("prompt.tokens"));
            let generated = read_ids(&base.join("generated.tokens"));
            let h_prompt = read_f32(&base.join("prefill.result_norm.f32"));
            let h_steps = read_f32(&base.join("steps.result_norm.f32"));
            let n = prompt.len();

            // The device follows the prompt in two steps (the second starts
            // from the hidden state the first kept).
            let cut = 100;
            rig.follow(0, &prompt[..cut], &h_prompt[..cut * d], true);
            rig.follow(cut as u32, &prompt[cut..], &h_prompt[cut * d..], true);

            let greedy = vec![SampleParams::GREEDY.row(0); LINKS];
            let fork_chain = fork
                .as_ref()
                .map(|dir| read_i32(&dir.join(prompt_name).join("mtp.chain.tokens.i32")));
            // The second chain starts after a window of WINDOW accepted rows.
            let starts = [(n, generated[0]), (n + WINDOW, generated[WINDOW])];
            let mut second = None;
            for (round, (pos0, token)) in starts.into_iter().enumerate() {
                if round == 1 {
                    // The window: the newest token and four more, all accepted.
                    let window_hidden = &h_steps[..WINDOW * d];
                    rig.follow(n as u32, &generated[..WINDOW], window_hidden, false);
                    // Its outcome: the block follows the accepted rows, and the
                    // hidden state of the last is the one the next chain starts
                    // from.
                    rig.commit(WINDOW as u32);
                }
                let want = &wanted[round];
                let got = rig.chain(pos0 as u32, token, &greedy);
                let want_tokens: Vec<u32> = want.iter().map(|l| l.token).collect();
                let mut worst = 0.0f64;
                let mut same_so_far = true;
                for (j, link) in want.iter().enumerate() {
                    if same_so_far {
                        worst = worst.max(rel(&got.outputs[j], &link.rows.hidden));
                    }
                    same_so_far &= got.drafts[j] == link.token;
                }
                // The logits of the last link, on the draft head's tokens.
                let last = &want[LINKS - 1].rows.logits;
                let want_logits: Vec<f32> =
                    draft_tokens.iter().map(|t| last[*t as usize]).collect();
                let logits = if same_so_far || got.drafts[..LINKS - 1] == want_tokens[..LINKS - 1] {
                    rel(&got.logits, &want_logits)
                } else {
                    f64::NAN
                };
                println!(
                    "{precision:?} {prompt_name} chain {round} at {pos0}: device {:?}, reference {:?}{}; \
                     link outputs rel L2 worst {worst:.2e}; logits of the last link rel L2 {logits:.2e}",
                    got.drafts,
                    want_tokens,
                    match (&fork_chain, round) {
                        (Some(f), 0) => format!(", fork {f:?}"),
                        _ => String::new(),
                    },
                );
                // The drafts are where the next kernels and the host read them.
                assert_eq!(got.record_tokens[0], token);
                assert_eq!(got.record_tokens[1..], got.drafts[..]);
                for (pick, draft) in got.picks.iter().zip(&got.drafts) {
                    assert_eq!((pick.token, pick.status), (*draft, 0));
                }
                assert_eq!(got.drafts[0], want_tokens[0], "the first draft is the reference's");
                let limit = if precision == Precision::Exact { 1.5e-2 } else { 6e-2 };
                assert!(worst < limit, "link outputs {worst:.2e} from the reference");
                if precision == Precision::Exact {
                    assert_eq!(got.drafts, want_tokens, "the chain is the reference's");
                }
                second = Some(got);
            }
            let second = second.expect("two chains");

            // A chain that its bar ends after the first link: the plans of the
            // links it did not make are not the next chain's (the next chain
            // read past its tables with them, 2026-11-17). The same chain again
            // is then the chain it was, bit for bit.
            rig.part.draft_ms();
            let links = rig.part.mtp_times().links;
            let even: DraftChances = std::array::from_fn(|_| 0.5);
            let (pos0, token) = ((n + WINDOW) as u32, generated[WINDOW]);
            let bars = vec![0.0, 1.0, 1.0, 1.0];
            let cut = rig.chain_with(pos0, token, &greedy, bars, Some(even));
            rig.part.draft_ms();
            assert_eq!(rig.part.mtp_times().links, links + 1, "the bar ended the chain");
            assert_eq!(cut.drafts[0], second.drafts[0]);
            let again = rig.chain(pos0, token, &greedy);
            assert_eq!(again.drafts, second.drafts, "the chain after a chain that was cut");
            for (a, b) in again.outputs.iter().zip(&second.outputs) {
                assert_eq!(bits(a), bits(b));
            }

            // With sampling a link's pick is the sampler's pick from its logits.
            let sampled = SampleParams {
                temperature: 1.0,
                top_k: 20,
                top_p: 0.95,
                seed: 77,
                ..SampleParams::GREEDY
            };
            let rows: Vec<SampleRow> =
                (0..1).map(|j| sampled.row((n + WINDOW + j) as u32)).collect();
            let got = rig.chain((n + WINDOW) as u32, generated[WINDOW], &rows);
            let host = sample_reference(&got.logits, |i| draft_tokens[i], &rows[0], &[], None).pick;
            assert_eq!(got.picks[0].token, host.token, "the sampled draft");
            assert_eq!(got.picks[0].value.to_bits(), host.value.to_bits());
            assert_eq!(got.picks[0].cand_token[..SAMPLE_CANDIDATES], host.cand_token[..]);
            println!(
                "{precision:?} {prompt_name}: sampled link picks {} (greedy would pick {})",
                got.picks[0].token, got.picks[0].cand_token[0]
            );
        }
        let times = rig.part.mtp_times();
        println!(
            "{precision:?}: {} links in {} chains: {:.3} ms per link on the device",
            times.links,
            times.steps,
            times.chain_ms / times.links.max(1) as f64
        );
    }
}
