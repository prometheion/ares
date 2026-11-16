//! The connections: accepted, served over HTTP/1.1, and closed when a client
//! hangs.
//!
//! A client that sends the head of a request slower than the timeout
//! (`--timeout`) is cut off by hyper's own timer, which also ends a
//! connection that waits longer than that for its next request; the body of a
//! request has the same time from the handler that reads it
//! (`routes::read_body`). A client that does not take an answer, so that one
//! write to its socket waits longer than the timeout, is cut off here
//! ([`Timed`]): its request is then cancelled in the engine, as for any
//! client that goes away. A connection that waits for the engine writes
//! nothing and is never cut: a reply that takes long is no hanging client.
//!
//! On the signal to stop, no connection is accepted any more, and the
//! connections that are open close as soon as their requests have answered.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpListener;
use tokio::time::Sleep;

/// Serves `router` on `listener` until `stop` completes, then waits until
/// every open connection has closed.
pub(crate) async fn listen(
    listener: TcpListener,
    router: Router,
    timeout: Duration,
    stop: impl Future<Output = ()>,
) -> io::Result<()> {
    let graceful = GracefulShutdown::new();
    let mut builder = hyper::server::conn::http1::Builder::new();
    builder.timer(TokioTimer::new()).header_read_timeout(timeout);
    tokio::pin!(stop);
    loop {
        let stream = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok((stream, _)) => stream,
                Err(error) => {
                    // Out of file descriptors, or a connection that was reset
                    // before it was taken: the next one may do.
                    tracing::warn!(%error, "a connection could not be accepted");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    continue;
                }
            },
            () = &mut stop => break,
        };
        let _ = stream.set_nodelay(true);
        let io = TokioIo::new(Timed::new(stream, timeout));
        let connection = builder.serve_connection(io, TowerToHyperService::new(router.clone()));
        let watched = graceful.watch(connection);
        tokio::spawn(async move {
            // A connection that fails has a client that is gone or broken;
            // its request was cancelled when its handler was dropped.
            let _ = watched.await;
        });
    }
    drop(listener);
    graceful.shutdown().await;
    Ok(())
}

/// A socket whose writes may not wait longer than a limit: a write that has
/// waited that long for room fails, and the connection closes.
pub(crate) struct Timed<S> {
    inner: S,
    limit: Duration,
    /// Since when a write waits, as the time it may still wait.
    waiting: Option<Pin<Box<Sleep>>>,
}

impl<S> Timed<S> {
    pub(crate) fn new(inner: S, limit: Duration) -> Self {
        Self { inner, limit, waiting: None }
    }

    /// The answer to a write that has to wait: the wait goes on, or it has
    /// lasted too long.
    fn wait<T>(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<T>> {
        let limit = self.limit;
        let waiting = self.waiting.get_or_insert_with(|| Box::pin(tokio::time::sleep(limit)));
        match waiting.as_mut().poll(cx) {
            Poll::Ready(()) => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                format!("the client took nothing of the answer for {} s", limit.as_secs()),
            ))),
            Poll::Pending => Poll::Pending,
        }
    }

    fn done<T>(
        &mut self,
        result: Poll<io::Result<T>>,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<T>> {
        match result {
            Poll::Ready(result) => {
                self.waiting = None;
                Poll::Ready(result)
            }
            Poll::Pending => self.wait(cx),
        }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for Timed<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_read(cx, buf)
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for Timed<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_write(cx, buf);
        this.done(result, cx)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_flush(cx);
        this.done(result, cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let result = Pin::new(&mut this.inner).poll_write_vectored(cx, bufs);
        this.done(result, cx)
    }

    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn a_write_that_waits_too_long_fails() {
        // A pipe of 64 bytes that nobody reads.
        let (writer, _reader) = tokio::io::duplex(64);
        let mut timed = Timed::new(writer, Duration::from_millis(100));
        timed.write_all(&[0; 64]).await.unwrap();
        let error = timed.write_all(&[0; 1]).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    }

    #[tokio::test]
    async fn a_write_that_waits_less_goes_through() {
        let (writer, mut reader) = tokio::io::duplex(64);
        let mut timed = Timed::new(writer, Duration::from_secs(5));
        timed.write_all(&[0; 64]).await.unwrap();
        let reading = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let mut buffer = vec![0; 128];
            let read = tokio::io::AsyncReadExt::read(&mut reader, &mut buffer).await.unwrap();
            // The reader stays open until the write is through.
            (read, reader)
        });
        timed.write_all(&[0; 10]).await.unwrap();
        assert_eq!(reading.await.unwrap().0, 64);
    }
}
