//! HTTP, the one module naming ureq: a download as a stream (setup.rs), a JSON post (llm.rs) and a form
//! with a file (whisper.rs, whisper.cpp's server); a local server started and stopped (serve, Served).
//! An answer with an error status is an error.
use anyhow::{Context, Result, bail};
use std::io::Read;
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
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("download {url}"))?;
    let size = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok()?.parse().ok())
        .unwrap_or(0);
    Ok((
        size,
        resp.into_body().into_with_config().limit(u64::MAX).reader(),
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
/// the answer's body.
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
