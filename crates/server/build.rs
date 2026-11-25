//! Names the build: the commit the binary is built from and the day it is
//! built on, for `ares --version`, the first line of the log and `/props`
//! (`ares_server::BUILD_INFO`): `ares <version> (<commit>, <date>)`.
//!
//! The commit is what `git rev-parse --short HEAD` says when the source is a
//! checkout of its own repository. A tree without `.git` (a `git archive`
//! snapshot, a copy made with rsync) names it in `ARES_BUILD_COMMIT`; without
//! either the commit is `unknown`. A tree that lies inside some other
//! repository is not taken for a checkout of that one.
//!
//! The day is the UTC date the build script ran, or the one of
//! `SOURCE_DATE_EPOCH` (seconds since 1970) for a build that is to come out
//! the same twice. The script runs again when the commit changes, not for
//! every edit, so a build of changes not yet committed carries the day the
//! commit was first built.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ARES_BUILD_COMMIT");
    println!("cargo:rerun-if-env-changed=SOURCE_DATE_EPOCH");

    let manifest =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    // The workspace's root: crates/server/..
    let root = manifest.join("../..");
    let commit = match checkout_commit(&root) {
        Some(commit) => commit,
        None => std::env::var("ARES_BUILD_COMMIT")
            .ok()
            .map(|commit| commit.trim().to_owned())
            .filter(|commit| !commit.is_empty())
            .unwrap_or_else(|| "unknown".to_owned()),
    };
    let version = std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION");
    println!("cargo:rustc-env=ARES_BUILD_INFO=ares {version} ({commit}, {})", build_date());
}

/// The short commit of `root` when it is the top of a git checkout, and
/// asks Cargo to run this script again when that checkout's HEAD moves.
fn checkout_commit(root: &Path) -> Option<String> {
    let top = git(root, &["rev-parse", "--show-toplevel"])?;
    let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if !same(Path::new(&top), root) {
        return None;
    }
    let commit = git(root, &["rev-parse", "--short", "HEAD"])?;
    // HEAD names the branch, the branch's file or the packed refs the commit,
    // and the log of HEAD changes with every commit, checkout and reset. A
    // path that does not exist would make Cargo run the script every time.
    let mut watched = vec!["HEAD".to_owned(), "logs/HEAD".to_owned(), "packed-refs".to_owned()];
    if let Some(branch) = git(root, &["symbolic-ref", "-q", "HEAD"]) {
        watched.push(branch);
    }
    for name in watched {
        if let Some(path) = git(root, &["rev-parse", "--git-path", &name]) {
            let path = root.join(path);
            if path.exists() {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
    Some(commit)
}

/// What `git` prints to `args` in `dir`, trimmed; `None` when it is not
/// there, fails or prints nothing.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?.trim().to_owned();
    (!text.is_empty()).then_some(text)
}

/// The day of the build, `YYYY-MM-DD` in UTC.
fn build_date() -> String {
    let seconds = std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or_else(|| {
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
        });
    let (year, month, day) = civil_date(seconds / 86_400);
    format!("{year:04}-{month:02}-{day:02}")
}

/// The date of the day `days` after 1970-01-01 in the proleptic Gregorian
/// calendar (Howard Hinnant's `civil_from_days`, for days from 1970 on).
fn civil_date(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z % 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 { shifted_month + 3 } else { shifted_month - 9 };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}
