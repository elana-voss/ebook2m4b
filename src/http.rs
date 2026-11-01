//! HTTP, the one module naming ureq: a download as a stream (setup.rs), a JSON post (llm.rs) and a form
//! with a file (whisper.rs, whisper.cpp's server); a local server started and stopped (serve, Served).
//! An answer with an error status is an error, a download that stalls too (a read timeout on the stream,
//! never a limit on the whole: an 8 GB download takes long).
use anyhow::{Context, Result, bail};
use std::io::{self, Read};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, sync_channel};
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
    if ok(health_url) {
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
        if ok(health_url) {
            return Ok(served);
        }
        if let Some(code) = served.0.try_wait()? {
            bail!("{name} ended at its start ({code})");
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    bail!("{name} did not start within {} s", wait.as_secs())
}

/// The body of `url` as a stream, and its size (0 when the server does not say).
pub fn get(url: &str) -> Result<(u64, impl Read)> {
    get_within(url, Duration::from_secs(60))
}

/// A body read in a thread: ureq's own limits cover the whole body, which an 8 GB download outlasts, so
/// the wait for each piece is timed here. A stalled thread ends with the program.
struct Stalling {
    pieces: Receiver<io::Result<Vec<u8>>>,
    piece: Vec<u8>,
    at: usize,
    stall: Duration,
    url: String,
}

impl Read for Stalling {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.at == self.piece.len() {
            self.piece = match self.pieces.recv_timeout(self.stall) {
                Ok(piece) => piece?,
                Err(RecvTimeoutError::Disconnected) => return Ok(0), // its end
                Err(RecvTimeoutError::Timeout) => {
                    let what = format!("{}: nothing for {} s", self.url, self.stall.as_secs());
                    return Err(io::Error::new(io::ErrorKind::TimedOut, what));
                }
            };
            self.at = 0;
        }
        let n = buf.len().min(self.piece.len() - self.at);
        buf[..n].copy_from_slice(&self.piece[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

/// get(), an error when the answer's head or the next piece of the body takes longer than `stall`.
fn get_within(url: &str, stall: Duration) -> Result<(u64, impl Read + use<>)> {
    let resp = ureq::get(url)
        .config()
        .timeout_recv_response(Some(stall))
        .build()
        .call()
        .with_context(|| format!("download {url}"))?;
    let size = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok()?.parse().ok())
        .unwrap_or(0);
    let mut body = resp.into_body().into_with_config().limit(u64::MAX).reader();
    let (send, pieces) = sync_channel(8);
    std::thread::spawn(move || {
        loop {
            let mut piece = vec![0; 64 * 1024];
            match body.read(&mut piece) {
                Ok(0) => break, // dropping `send` tells the reader it is the end
                Ok(n) => {
                    piece.truncate(n);
                    if send.send(Ok(piece)).is_err() {
                        break; // the reader is gone
                    }
                }
                Err(e) => {
                    let _ = send.send(Err(e)); // with the reader gone no one is left to tell
                    break;
                }
            }
        }
    });
    Ok((
        size,
        Stalling {
            pieces,
            piece: Vec::new(),
            at: 0,
            stall,
            url: url.to_string(),
        },
    ))
}

/// Whether `url` answers a GET with success within a second.
pub fn ok(url: &str) -> bool {
    ureq::get(url)
        .config()
        .timeout_global(Some(Duration::from_secs(1)))
        .build()
        .call()
        .is_ok()
}

/// POSTs `body` (JSON) to `url`, with `Authorization: Bearer <key>` when given; the answer's body.
pub fn post_json(url: &str, body: &str, key: Option<&str>, timeout: Duration) -> Result<String> {
    let mut req = ureq::post(url)
        .config()
        .timeout_global(Some(timeout))
        .build()
        .header("Content-Type", "application/json");
    if let Some(k) = key {
        req = req.header("Authorization", &format!("Bearer {k}"));
    }
    let mut resp = req.send(body).with_context(|| format!("POST {url}"))?;
    Ok(resp.body_mut().read_to_string()?)
}

/// POSTs a multipart form to `url`: the text `fields` and one file (`name`, its file name, its bytes);
/// the answer's body. Windows' whisper.cpp server only.
#[cfg(windows)]
pub fn post_form(
    url: &str,
    fields: &[(&str, &str)],
    file: (&str, &str, &[u8]),
    timeout: Duration,
) -> Result<String> {
    const BOUNDARY: &str = "ebook2m4b-form-7f3a9c";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    let (name, file_name, bytes) = file;
    body.extend_from_slice(
        format!(
            "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{BOUNDARY}--\r\n").as_bytes());
    let mut resp = ureq::post(url)
        .config()
        .timeout_global(Some(timeout))
        .build()
        .header(
            "Content-Type",
            &format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .send(&body[..])
        .with_context(|| format!("POST {url}"))?;
    Ok(resp.body_mut().read_to_string()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_download_that_stalls_is_an_error_naming_the_url() {
        use std::io::Write;
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let _ = s.read(&mut [0; 1024]); // the request; its content does not matter
            let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc");
            std::thread::sleep(Duration::from_secs(30)); // open, sending nothing
        });
        let url = format!("http://127.0.0.1:{port}/big.bin");
        let (size, mut body) = get_within(&url, Duration::from_millis(200)).unwrap();
        assert_eq!(size, 10);
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(body.read_to_end(&mut Vec::new()).map_err(|e| e.to_string()));
        });
        let err = rx.recv_timeout(Duration::from_secs(5)).expect("no hang").unwrap_err();
        assert!(err.contains("big.bin"), "{err}");
    }

    #[test]
    fn no_server_starts_where_another_answers() {
        use std::io::Write;
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
