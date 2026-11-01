//! A local server's process: started once it answers, stopped when dropped (llm.rs's and whisper.rs's).
use crate::setup::http;
use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// A server started by serve(): killed, and waited for, when dropped.
pub struct Served(Child);

impl Drop for Served {
    fn drop(&mut self) {
        // it may have ended already; nothing is left to stop then
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// `exe` started with `args`, once `health_url` answers a GET (asked every half second, at most `wait`).
/// Refused while something already answers there (one left from a run that was killed would answer for
/// ours, and dropping ours would stop nothing).
pub fn serve(exe: &Path, args: &[String], health_url: &str, wait: Duration) -> Result<Served> {
    let name = exe
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    if http::ok(health_url) {
        bail!("a server already answers at {health_url} (a {name} left running?); stop it first");
    }
    let child = Command::new(exe)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| exe.display().to_string())?;
    let mut served = Served(child);
    let t = Instant::now();
    while t.elapsed() < wait {
        if http::ok(health_url) {
            return Ok(served);
        }
        if let Some(code) = served.0.try_wait()? {
            bail!("{name} ended at its start ({code})");
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    bail!("{name} did not start within {} s", wait.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_server_starts_where_another_answers() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                let _ = s.read(&mut [0; 1024]); // the request; its content does not matter
                let _ = s.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
            }
        });
        let url = format!("http://127.0.0.1:{port}/health");
        let err = serve(
            Path::new("no-such-server.exe"),
            &[],
            &url,
            Duration::from_secs(1),
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("already answers"), "{err}");
    }
}
