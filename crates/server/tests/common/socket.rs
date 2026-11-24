//! HTTP/1.1 as a client speaks it on a plain socket: one request on its own
//! connection, which the server closes after the answer.

use std::io::Read;
use std::net::TcpStream;

use serde_json::Value;

/// The bytes of a request with `body` (which may be empty) as JSON.
pub fn request_bytes(method: &str, path: &str, body: &str) -> Vec<u8> {
    format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// One answer of the server.
pub struct Reply {
    pub status: u16,
    /// The status line and the headers, as they came.
    pub head: String,
    /// The body, without the chunked transfer coding.
    pub body: String,
}

impl Reply {
    /// The answer in `raw`, everything the server sent on the connection.
    pub fn parse(raw: &[u8]) -> Self {
        let split = find(raw, b"\r\n\r\n").expect("a complete header");
        let head = String::from_utf8_lossy(&raw[..split]).into_owned();
        let status = head.split(' ').nth(1).and_then(|s| s.parse().ok()).expect("a status line");
        let body = &raw[split + 4..];
        let chunked = head.to_ascii_lowercase().contains("transfer-encoding: chunked");
        let body = if chunked { dechunk(body) } else { body.to_vec() };
        Self { status, head, body: String::from_utf8(body).expect("the body is valid UTF-8") }
    }

    pub fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", self.body))
    }

    /// The `data:` payloads of a streamed answer and whether `[DONE]` ended it.
    pub fn events(&self) -> (Vec<Value>, bool) {
        let stream = super::parse_sse(&self.body);
        (stream.chunks, stream.done)
    }

    /// Whether the head names `content_type`.
    pub fn is(&self, content_type: &str) -> bool {
        self.head.to_ascii_lowercase().contains(&format!("content-type: {content_type}"))
    }
}

pub fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Removes the chunked transfer coding; a stream that was cut ends where the
/// last whole chunk ends.
fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(line_end) = find(raw, b"\r\n") {
        let size = std::str::from_utf8(&raw[..line_end])
            .ok()
            .and_then(|s| usize::from_str_radix(s.trim(), 16).ok());
        let Some(size) = size else { break };
        let start = line_end + 2;
        if size == 0 || raw.len() < start + size {
            break;
        }
        out.extend_from_slice(&raw[start..start + size]);
        raw = &raw[(start + size + 2).min(raw.len())..];
    }
    out
}

/// Reads from `stream` until `count` events of a streamed answer have come.
pub fn read_events(stream: &mut TcpStream, count: usize) -> Vec<u8> {
    let mut seen = Vec::new();
    let mut buffer = [0u8; 4096];
    while String::from_utf8_lossy(&seen).matches("data: ").count() < count {
        let n = stream.read(&mut buffer).unwrap();
        assert!(n > 0, "the stream ended early: {}", String::from_utf8_lossy(&seen));
        seen.extend_from_slice(&buffer[..n]);
    }
    seen
}

/// The text of a streamed chat or completion: its deltas put together, the
/// reasoning of a model that thinks (its own field) before the content.
pub fn streamed_text(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|chunk| {
            let choice = chunk["choices"].get(0)?;
            let delta = &choice["delta"];
            Some(
                [
                    delta["reasoning_content"].as_str(),
                    delta["content"].as_str(),
                    choice["text"].as_str(),
                ]
                .into_iter()
                .flatten()
                .collect::<String>(),
            )
        })
        .collect()
}
