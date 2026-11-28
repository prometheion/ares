//! The restart in place: a new image of the server in the same process, after
//! a model's engine was found stuck in a step (`Models::restart_asked`).
//!
//! A step that does not come back holds the memory of the devices with
//! threads that wait in it for good, and the driver may not answer for any
//! more, so no unload gives that memory back. The server answers every request
//! (503 with the reason), then replaces its image with `execve` of
//! `/proc/self/exe`, with the same arguments and environment: the kernel ends
//! every thread of the old image and frees its memory, the page-locked RAM
//! among it, and with the driver's files, which the exec closes, the memory of
//! the devices. The listening socket stays open across the exec and the new
//! image takes it over ([`LISTEN_FD`]), so the process keeps its PID and the
//! port: a supervisor (systemd, llama-swap) sees a short outage, while new
//! connections wait in the socket's backlog for the new image to load the
//! model.
//!
//! When the exec cannot be done, the process exits with [`EXIT_RESTART`] for
//! its supervisor to start it again. A thread that sits in the driver without a
//! way out (state D) holds an exec as it holds an exit, until the driver lets
//! it go: the server waits a moment for it, then exits that way.

use std::ffi::CString;
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::time::{Duration, Instant};

/// The variable of the environment that tells a new image the descriptor of
/// the listening socket it takes over from the image before.
pub const LISTEN_FD: &str = "ARES_LISTEN_FD";
/// The variable that tells a new image why the image before restarted.
pub const RESTARTED: &str = "ARES_RESTARTED";
/// The exit code of a server that found an engine stuck and could not
/// restart in place: its supervisor is to start it again.
pub const EXIT_RESTART: i32 = 75;

/// How long a thread in the driver without a way out is waited for before
/// the server exits instead of restarting in place.
const UNINTERRUPTIBLE_PATIENCE: Duration = Duration::from_secs(5);

/// The listening socket the image before left to this one, when this image
/// is a restart in place ([`LISTEN_FD`]): it is closed on any later exec of
/// another program again, and serves as it did. `None` when there is none, or
/// when the variable names no listening socket of this process (the port is
/// bound then, as at any start).
pub fn inherited() -> Option<std::net::TcpListener> {
    let value = std::env::var(LISTEN_FD).ok()?;
    let Ok(fd) = value.parse::<RawFd>() else {
        tracing::warn!("{LISTEN_FD}={value} names no descriptor: binding the port");
        return None;
    };
    if fd <= 2 || !listening(fd) {
        tracing::warn!("{LISTEN_FD}={value} names no listening socket: binding the port");
        return None;
    }
    // SAFETY: `fd` is an open listening socket that nothing else in this
    // image owns: the image before left it to this one.
    let listener = unsafe { std::net::TcpListener::from_raw_fd(fd) };
    // SAFETY: `fd` is open; the flag only closes it on a later exec.
    unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
    if let Err(error) = listener.set_nonblocking(true) {
        tracing::warn!(
            "the listening socket of the image before ({fd}): {error}: binding the port"
        );
        return None;
    }
    match std::env::var(RESTARTED) {
        Ok(reason) => tracing::warn!(
            fd,
            "this image of the server replaces one in the same process that restarted in place \
             ({reason}): its listening socket is taken over"
        ),
        Err(_) => tracing::info!(fd, "the listening socket of the image before is taken over"),
    }
    Some(listener)
}

/// Whether `fd` is a listening socket of the internet.
fn listening(fd: RawFd) -> bool {
    let option = |name: libc::c_int| {
        let mut value: libc::c_int = 0;
        let mut len = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
        // SAFETY: `value` and `len` are valid for the call; an `fd` that is
        // not a socket makes it fail.
        let done = unsafe {
            libc::getsockopt(fd, libc::SOL_SOCKET, name, (&raw mut value).cast(), &mut len)
        };
        (done == 0).then_some(value)
    };
    option(libc::SO_ACCEPTCONN) == Some(1)
        && matches!(option(libc::SO_DOMAIN), Some(libc::AF_INET | libc::AF_INET6))
}

/// A second descriptor of the listening socket, which outlives the server's
/// own when it stops accepting: the socket goes on listening, and what
/// connects meanwhile waits in its backlog for the new image.
pub fn keep(listener: &impl AsRawFd) -> Result<OwnedFd, String> {
    // SAFETY: duplicating an open descriptor; the copy closes on an exec
    // until the restart says otherwise.
    let fd = unsafe { libc::fcntl(listener.as_raw_fd(), libc::F_DUPFD_CLOEXEC, 3) };
    if fd < 0 {
        return Err(format!(
            "cannot keep the listening socket: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: `fd` was just made and is owned by nobody else.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Replaces the image of the process by a new one of the same program with
/// the same arguments and environment, which takes the listening socket over;
/// exits with [`EXIT_RESTART`] when that cannot be done. Everything worth
/// writing is written and every request is answered by now.
pub fn restart_in_place(listener: OwnedFd, reason: &str) -> ! {
    let held = settled_threads(UNINTERRUPTIBLE_PATIENCE);
    if !held.is_empty() {
        tracing::error!(
            threads = %held.join(", "),
            "threads sit in the driver without a way out (state D), which would hold an exec as \
             they hold an exit: exiting with {EXIT_RESTART} for the supervisor to start the \
             server again"
        );
        crate::exit_now(EXIT_RESTART);
    }
    let fd = listener.into_raw_fd();
    let Some((argv, envp)) = image(fd, reason) else {
        tracing::error!(
            "the arguments or the environment hold a NUL byte: exiting with {EXIT_RESTART}"
        );
        crate::exit_now(EXIT_RESTART);
    };
    mark_close_on_exec();
    // SAFETY: `fd` is open; it alone of this image's descriptors stays open
    // across the exec.
    unsafe { libc::fcntl(fd, libc::F_SETFD, 0) };
    tracing::error!(
        fd,
        "restarting in place: a new image of the server with the same arguments, the listening \
         socket kept"
    );
    let argv_ptrs: Vec<*const libc::c_char> =
        argv.iter().map(|arg| arg.as_ptr()).chain(std::iter::once(std::ptr::null())).collect();
    let envp_ptrs: Vec<*const libc::c_char> =
        envp.iter().map(|var| var.as_ptr()).chain(std::iter::once(std::ptr::null())).collect();
    // SAFETY: the path, the arguments and the environment are NUL-terminated
    // strings in arrays that end with a null pointer, alive across the call.
    unsafe { libc::execve(c"/proc/self/exe".as_ptr(), argv_ptrs.as_ptr(), envp_ptrs.as_ptr()) };
    let error = std::io::Error::last_os_error();
    tracing::error!(
        "the restart in place failed ({error}): exiting with {EXIT_RESTART} for the supervisor \
         to start the server again"
    );
    crate::exit_now(EXIT_RESTART)
}

/// The arguments of this image and its environment for the next one, which
/// is told the listening socket `fd` and why it starts.
fn image(fd: RawFd, reason: &str) -> Option<(Vec<CString>, Vec<CString>)> {
    let argv: Option<Vec<CString>> =
        std::env::args_os().map(|arg| CString::new(arg.as_bytes()).ok()).collect();
    let ours = |name: &std::ffi::OsStr| name == LISTEN_FD || name == RESTARTED;
    let mut envp: Vec<CString> = std::env::vars_os()
        .filter(|(name, _)| !ours(name))
        .map(|(name, value)| CString::new([name.as_bytes(), b"=", value.as_bytes()].concat()).ok())
        .collect::<Option<_>>()?;
    envp.push(CString::new(format!("{LISTEN_FD}={fd}")).ok()?);
    let reason = reason.replace(['\0', '\n'], " ");
    envp.push(CString::new(format!("{RESTARTED}={reason}")).ok()?);
    Some((argv?, envp))
}

/// Marks every descriptor above the standard three to be closed by the exec,
/// whatever its owner opened it with: the driver's files go with the old image
/// and the memory of the devices with them.
fn mark_close_on_exec() {
    // SAFETY: close_range with CLOSE_RANGE_CLOEXEC only sets a flag.
    let marked =
        unsafe { libc::syscall(libc::SYS_close_range, 3u32, u32::MAX, libc::CLOSE_RANGE_CLOEXEC) }
            == 0;
    if marked {
        return;
    }
    // A kernel before 5.11: one by one.
    let Ok(entries) = std::fs::read_dir("/proc/self/fd") else { return };
    for fd in entries.flatten().filter_map(|e| e.file_name().to_str()?.parse::<RawFd>().ok()) {
        if fd > 2 {
            // SAFETY: setting a flag of a descriptor; one that closed since
            // makes the call fail, which changes nothing.
            unsafe { libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC) };
        }
    }
}

/// The threads of this process that sit in the kernel without a way out
/// (state D), once `patience` has given them the chance to leave; their ids
/// and names.
fn settled_threads(patience: Duration) -> Vec<String> {
    let begun = Instant::now();
    loop {
        let held = uninterruptible();
        if held.is_empty() || begun.elapsed() >= patience {
            return held;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn uninterruptible() -> Vec<String> {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else { return Vec::new() };
    tasks
        .flatten()
        .filter_map(|task| {
            let stat = std::fs::read_to_string(task.path().join("stat")).ok()?;
            // `tid (name) state ...`: the name may hold spaces and parentheses.
            let (head, rest) = stat.rsplit_once(')')?;
            let state = rest.trim_start().chars().next()?;
            let name = head.split_once('(').map_or("", |(_, name)| name);
            (state == 'D').then(|| format!("{} ({name})", task.file_name().to_string_lossy()))
        })
        .collect()
}
