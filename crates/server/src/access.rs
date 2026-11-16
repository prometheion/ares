//! Who may use the server: API keys, the web pages that may call it from a
//! browser (CORS), and the names and origins a server on a local address
//! answers to.
//!
//! One layer over every route ([`guard`]), which looks at a request in this
//! order:
//!
//! 1. **The name the request was sent to** (`Host`). A server that listens on
//!    a local address and has no key answers only to its local names:
//!    `localhost` and the names under it, and addresses written as numbers. A
//!    web page that rebinds a name of its own to 127.0.0.1 sends that name, and
//!    gets 403. `--allowed-hosts` adds names, or with `*` lifts the check. A
//!    request without a `Host` (only HTTP/1.0 sends none) cannot come from a
//!    browser and passes.
//! 2. **A browser's preflight.** An `OPTIONS` request from an origin of
//!    `--cors-origins` is answered here, without a key: a browser sends none
//!    with it.
//! 3. **The page a request comes from** (`Origin`). Without a key, a request
//!    that a browser sent from a page of another site gets 403, so that a page
//!    cannot make a local server work for it. Pages of this machine
//!    (`localhost`, a loopback address), pages of the server's own name and of
//!    an allowed host, and the origins of `--cors-origins` pass. A program that is no browser sends
//!    no `Origin` and passes.
//! 4. **The key.** With keys, every route but `/health` wants one of them, as
//!    `Authorization: Bearer <key>` or as `x-api-key: <key>` (Anthropic's
//!    header, which its SDKs send). It is compared in constant time with every
//!    key. Without, 401 in the error shape of the route's API.
//!
//! With a key the key decides: the name and the page are not checked unless
//! `--allowed-hosts` names hosts (the server's start sets that, see
//! `crate::Access`). An answer to an origin of `--cors-origins` carries the
//! headers that let the page read it, errors included.

use std::net::IpAddr;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::error::{Anthropic, ApiError};

/// Who may use the server. [`Access::default`] lets everyone in: what a
/// server that is given no keys, no hosts and no origins does on an address
/// beyond this machine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Access {
    /// The keys a request must carry one of; none: no key is asked for.
    pub keys: Vec<String>,
    /// The names in a request's `Host` that the server answers to besides its
    /// local ones; `None`: every name.
    pub hosts: Option<Vec<String>>,
    /// The origins whose pages may call the server from a browser and read
    /// its answers; `*` stands for every origin.
    pub origins: Vec<String>,
}

impl Access {
    /// Whether a request needs a key.
    pub fn locked(&self) -> bool {
        !self.keys.is_empty()
    }

    fn any_origin(&self) -> bool {
        self.origins.iter().any(|origin| origin == "*")
    }

    /// Whether pages of `origin` may call the server from a browser.
    fn cors(&self, origin: &str) -> bool {
        self.any_origin() || self.origins.iter().any(|known| known == origin)
    }

    /// Whether the server answers to `host` (a name or an address, without
    /// the port).
    fn host_allowed(&self, host: &str) -> bool {
        match &self.hosts {
            None => true,
            Some(listed) => local_host(host) || listed.iter().any(|name| same_name(name, host)),
        }
    }

    /// Whether the page of `origin` may call a server without keys that was
    /// asked as `host`.
    fn origin_allowed(&self, origin: &str, host: Option<&str>) -> bool {
        if self.cors(origin) {
            return true;
        }
        let Some(own) = origin_host(origin) else { return false };
        loopback_host(own)
            || host.is_some_and(|host| same_name(host, own))
            || self.hosts.iter().flatten().any(|name| same_name(name, own))
    }

    /// Whether `headers` carry one of the keys. Every key is compared in
    /// full, so that the time it takes says nothing about any of them.
    fn admits(&self, headers: &HeaderMap) -> bool {
        let bearer = headers
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| {
                let (scheme, key) = value.trim().split_once(' ')?;
                scheme.eq_ignore_ascii_case("bearer").then(|| key.trim())
            });
        let named = headers.get("x-api-key").and_then(|value| value.to_str().ok());
        let mut found = false;
        for given in [bearer, named].into_iter().flatten() {
            for key in &self.keys {
                found |= same_key(given.as_bytes(), key.as_bytes());
            }
        }
        found
    }
}

/// Whether two keys are the same, in a time that depends on their lengths
/// alone.
fn same_key(given: &[u8], key: &[u8]) -> bool {
    if given.len() != key.len() {
        return false;
    }
    let difference = given.iter().zip(key).fold(0u8, |acc, (a, b)| acc | (a ^ b));
    std::hint::black_box(difference) == 0
}

/// Whether two host names are the same: names do not tell case apart.
fn same_name(a: &str, b: &str) -> bool {
    a.trim_end_matches('.').eq_ignore_ascii_case(b.trim_end_matches('.'))
}

/// Whether `host` is a name no web site can make its own: `localhost` and
/// the names under it, and an address written as numbers.
fn local_host(host: &str) -> bool {
    let host = host.trim_end_matches('.');
    let bare = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(host);
    bare.parse::<IpAddr>().is_ok()
        || host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
}

/// Whether `host` is this machine: `localhost` and the names under it, and
/// the loopback addresses. A page served from there is a local page.
fn loopback_host(host: &str) -> bool {
    let host = host.trim_end_matches('.');
    let bare = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(host);
    match bare.parse::<IpAddr>() {
        Ok(address) => address.is_loopback(),
        Err(_) => {
            host.eq_ignore_ascii_case("localhost")
                || host.to_ascii_lowercase().ends_with(".localhost")
        }
    }
}

/// The host of a `Host` header or of an authority: without its port, an
/// IPv6 address in its brackets.
fn host_of(authority: &str) -> &str {
    if authority.starts_with('[') {
        return authority.find(']').map_or(authority, |end| &authority[..=end]);
    }
    authority.rsplit_once(':').map_or(authority, |(host, _)| host)
}

/// The host of an origin (`https://chat.example.com:8443`); `None` for one
/// that has none (`null`).
fn origin_host(origin: &str) -> Option<&str> {
    let (_, rest) = origin.split_once("://")?;
    let authority = rest.split('/').next().unwrap_or(rest);
    let authority = authority.rsplit_once('@').map_or(authority, |(_, after)| after);
    Some(host_of(authority)).filter(|host| !host.is_empty())
}

/// Whether a route answers in Anthropic's shape of an error.
fn anthropic_route(path: &str) -> bool {
    path == "/v1/messages" || path.starts_with("/v1/messages/")
}

fn refusal(path: &str, error: ApiError) -> Response {
    if anthropic_route(path) { Anthropic(error).into_response() } else { error.into_response() }
}

/// The headers that let the page of `origin` read an answer.
fn allow(response: &mut Response, access: &Access, origin: &HeaderValue) {
    let headers = response.headers_mut();
    let value = if access.any_origin() { HeaderValue::from_static("*") } else { origin.clone() };
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
    headers.append(header::VARY, HeaderValue::from_static("Origin"));
    headers.insert(
        header::ACCESS_CONTROL_EXPOSE_HEADERS,
        HeaderValue::from_static("retry-after, x-ares-max-tokens-limit"),
    );
}

/// The answer to a browser's preflight from an origin that may call the
/// server: the methods the routes take, and the headers the browser asks for
/// (a key, the type of the body, Anthropic's version), however it names them.
fn preflight(access: &Access, origin: &HeaderValue, request: &Request) -> Response {
    let mut response = StatusCode::NO_CONTENT.into_response();
    allow(&mut response, access, origin);
    let headers = response.headers_mut();
    headers.insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST"));
    if let Some(asked) = request.headers().get(header::ACCESS_CONTROL_REQUEST_HEADERS) {
        headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, asked.clone());
    }
    headers.insert(header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("600"));
    response
}

/// The layer over every route: see the module documentation.
pub(crate) async fn guard(
    State(access): State<Arc<Access>>,
    request: Request,
    next: Next,
) -> Response {
    let path = request.uri().path().to_owned();
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .or_else(|| request.uri().authority().map(|authority| authority.as_str()))
        .map(|authority| host_of(authority).to_owned());
    if let Some(host) = &host
        && !access.host_allowed(host)
    {
        return refusal(&path, ApiError::forbidden_host(host));
    }

    let origin = request.headers().get(header::ORIGIN).cloned();
    let text = origin.as_ref().map(|o| o.to_str().unwrap_or("").to_owned());
    let cors = text.as_deref().is_some_and(|origin| access.cors(origin));
    if let Some(origin) = &origin
        && cors
        && request.method() == Method::OPTIONS
        && request.headers().contains_key(header::ACCESS_CONTROL_REQUEST_METHOD)
    {
        return preflight(&access, origin, &request);
    }
    if let Some(text) = &text
        && !access.locked()
        && !access.origin_allowed(text, host.as_deref())
    {
        return refusal(&path, ApiError::forbidden_origin(text));
    }

    let mut response = if access.locked() && path != "/health" && !access.admits(request.headers())
    {
        refusal(&path, ApiError::unauthorized())
    } else {
        next.run(request).await
    };
    if let Some(origin) = &origin
        && cors
    {
        allow(&mut response, &access, origin);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_names_are_localhost_and_addresses() {
        for host in ["localhost", "LOCALHOST", "app.localhost", "127.0.0.1", "10.0.0.7", "[::1]"] {
            assert!(local_host(host), "{host}");
        }
        for host in ["example.com", "localhost.example.com", "127.0.0.1.nip.io", "[::1"] {
            assert!(!local_host(host), "{host}");
        }
    }

    #[test]
    fn local_pages_are_pages_of_this_machine() {
        for host in ["localhost", "app.localhost", "127.0.0.1", "127.1.2.3", "[::1]"] {
            assert!(loopback_host(host), "{host}");
        }
        for host in ["10.0.0.7", "192.168.1.50", "example.com", "[::2]"] {
            assert!(!loopback_host(host), "{host}");
        }
    }

    #[test]
    fn the_host_of_an_authority_and_of_an_origin() {
        assert_eq!(host_of("localhost:8000"), "localhost");
        assert_eq!(host_of("[::1]:8000"), "[::1]");
        assert_eq!(host_of("[::1]"), "[::1]");
        assert_eq!(host_of("example.com"), "example.com");
        assert_eq!(origin_host("http://localhost:3000"), Some("localhost"));
        assert_eq!(origin_host("https://chat.example.com"), Some("chat.example.com"));
        assert_eq!(origin_host("http://[::1]:5173"), Some("[::1]"));
        assert_eq!(origin_host("null"), None);
    }

    #[test]
    fn keys_are_equal_only_in_full() {
        assert!(same_key(b"secret", b"secret"));
        assert!(!same_key(b"secret", b"secreT"));
        assert!(!same_key(b"secret", b"secret2"));
        assert!(!same_key(b"", b"secret"));
    }
}
