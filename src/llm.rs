//! A language model asked for answers in a fixed JSON shape, local or remote (llm.py's port).
//!
//! Both speak the OpenAI chat API. Local (the default): llama.cpp's server and Gemma 4 E4B from setup.rs,
//! on the NVIDIA GPU (Vulkan on Windows, CUDA on Linux; another card: its first Vulkan device), started at the first question and stopped
//! by close(). Remote: a file llm.json in the home folder (setup::home) holding
//! {"url": "https://.../v1", "model": "<name>", "key_env": "<VARIABLE>"}, the key read from that
//! environment variable, never written down.
//! The answer is held to a JSON schema (llama.cpp grammar, OpenAI structured output); temperature 0.
//! Answers are kept in work/llm-cache.json under llm.py's key (SHA-1 of Python's json.dumps of
//! [model, system, user, schema]), so t2a.py and this program share them and asking again costs nothing.
//! Trap: the key needs Python's JSON spacing and its \u escapes (py_json) and serde_json's
//! preserve_order, so objects keep the order they were written in.
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

const PORT: u16 = 8932;

pub enum Place {
    /// llama.cpp's server (its program) and the model file
    Local { server: PathBuf, gguf: PathBuf },
    Remote {
        url: String,
        model: String,
        key_env: String,
    },
}

/// Remote when home\llm.json says so, else local with what setup found.
pub fn place(home: &Path, found: &BTreeMap<String, PathBuf>) -> Result<Place> {
    let cfg = home.join("llm.json");
    if let Some(s) = read_if_there(&cfg)? {
        let v: Value = serde_json::from_str(&s).with_context(|| cfg.display().to_string())?;
        let field = |k: &str| {
            v[k].as_str()
                .map(str::to_string)
                .with_context(|| format!("{} has no \"{k}\"", cfg.display()))
        };
        return Ok(Place::Remote {
            url: field("url")?,
            model: field("model")?,
            key_env: field("key_env")?,
        });
    }
    let dir = found
        .get("llama-server")
        .context("llama.cpp is not set up")?;
    let gguf = found
        .get("llm")
        .context("the language model is not set up")?;
    Ok(Place::Local {
        server: dir.join(crate::os::exe(crate::os::THIS, "llama-server")),
        gguf: gguf.clone(),
    })
}

pub struct Llm {
    place: Place,
    name: String,
    server: Option<crate::http::Served>,
    cache_file: PathBuf,
    cache: Map<String, Value>,
}

impl Llm {
    pub fn new(place: Place, cache_file: PathBuf) -> Result<Llm> {
        let name = match &place {
            Place::Local { gguf, .. } => gguf
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into(),
            Place::Remote { model, .. } => model.clone(),
        };
        let cache = match read_if_there(&cache_file)? {
            Some(s) => {
                serde_json::from_str(&s).with_context(|| cache_file.display().to_string())?
            }
            None => Map::new(), // no answers kept yet
        };
        Ok(Llm {
            place,
            name,
            server: None,
            cache_file,
            cache,
        })
    }

    /// The answer to `user` under `system`, in the shape of `schema`; from the cache when asked before.
    pub fn ask(&mut self, system: &str, user: &str, schema: &Value) -> Result<Value> {
        let key = json!([self.name, system, user, schema]);
        let key = crate::hash::sha1(py_json(&key, true));
        if let Some(v) = self.cache.get(&key) {
            return Ok(v.clone());
        }
        crate::heat::wait();
        let got = self.post(system, user, schema)?;
        self.cache.insert(key, got.clone());
        if let Some(dir) = self.cache_file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(
            &self.cache_file,
            py_json(&Value::Object(self.cache.clone()), true),
        )
        .with_context(|| self.cache_file.display().to_string())?;
        Ok(got)
    }

    fn post(&mut self, system: &str, user: &str, schema: &Value) -> Result<Value> {
        let (url, key) = self.endpoint()?;
        let body = json!({
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0, "model": self.name,
            "response_format": {"type": "json_schema",
                "json_schema": {"name": "answer", "strict": true, "schema": schema}},
        });
        let got = crate::http::post_json(
            &format!("{url}/chat/completions"),
            &body.to_string(),
            key.as_deref(),
            Duration::from_secs(600),
        )?;
        let got: Value = serde_json::from_str(&got).context("the model's answer is no JSON")?;
        let content = got["choices"][0]["message"]["content"]
            .as_str()
            .context("the model's answer has no content")?;
        serde_json::from_str(content).context("the model's content is no JSON")
    }

    /// The API's base URL and key; a local server is started first.
    fn endpoint(&mut self) -> Result<(String, Option<String>)> {
        match &self.place {
            Place::Remote { url, key_env, .. } => {
                let key =
                    std::env::var(key_env).with_context(|| format!("{key_env} is not set"))?;
                Ok((url.trim_end_matches('/').to_string(), Some(key)))
            }
            Place::Local { server, gguf } => {
                if self.server.is_none() {
                    self.server = Some(start(server, gguf)?);
                }
                Ok((format!("http://127.0.0.1:{PORT}/v1"), None))
            }
        }
    }

    /// Stops a local server (the GPU's memory to the voice); answers stay in the cache.
    pub fn close(&mut self) {
        self.server = None;
    }
}

/// A file's text; None when there is no such file (any other failure is an error: a locked cache
/// read as empty would be overwritten with the next answer).
fn read_if_there(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| path.display().to_string()),
    }
}

/// llama.cpp's server on the GPU, answering on PORT (http::serve, at most 120 s).
fn start(exe: &Path, gguf: &Path) -> Result<crate::http::Served> {
    let out = Command::new(exe)
        .arg("--list-devices")
        .output()
        .with_context(|| exe.display().to_string())?;
    let listed = String::from_utf8_lossy(&out.stdout);
    let device = |want: &str| {
        regex::Regex::new(&format!(r"(?m)^\s*(\S+): {want}"))
            .expect("a valid pattern")
            .captures(&listed)
            .map(|c| c[1].to_string())
    };
    // Vulkan lists the integrated AMD GPU too
    let Some(device) = device("NVIDIA").or_else(|| device("")) else {
        bail!("llama.cpp sees no GPU; the language model runs on the GPU only");
    };
    let args = [
        "-m",
        &gguf.to_string_lossy(),
        "--port",
        &PORT.to_string(),
        "--device",
        &device,
        "-ngl",
        "999",
        "-c",
        "4096",
        "-t",
        "4",
        "--jinja",
        "--reasoning-budget",
        "0",
        "--load-mode", // not mapped: mapped, the file stayed in RAM twice over (14 GB)
        "none",
    ]
    .map(String::from);
    let health = format!("http://127.0.0.1:{PORT}/health");
    crate::http::serve(exe, &args, &health, Duration::from_secs(120))
}

/// JSON as Python's json.dumps writes it: ", " and ": " between items; ascii: non-ASCII as \uXXXX
/// (ensure_ascii).
pub fn py_json(v: &Value, ascii: bool) -> String {
    let mut out = String::new();
    write_py(v, ascii, &mut out);
    out
}

fn write_py(v: &Value, ascii: bool, out: &mut String) {
    match v {
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(x, ascii, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py(&Value::String(k.clone()), ascii, out);
                out.push_str(": ");
                write_py(x, ascii, out);
            }
            out.push('}');
        }
        Value::String(_) if ascii => {
            for c in v.to_string().chars() {
                if c.is_ascii() && c != '\x7f' {
                    // Python escapes DEL too
                    out.push(c);
                } else {
                    for u in c.encode_utf16(&mut [0; 2]) {
                        out.push_str(&format!("\\u{u:04x}"));
                    }
                }
            }
        }
        _ => out.push_str(&v.to_string()),
    }
}

/// A chat server for tests: answers each request with reply(its body as JSON) as the content, and keeps
/// each request (headers and body).
#[cfg(test)]
pub mod fake {
    use serde_json::Value;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    pub type Seen = Arc<Mutex<Vec<(String, Value)>>>;

    /// Its base URL and the requests seen.
    pub fn server(reply: impl Fn(&Value) -> Value + Send + 'static) -> (String, Seen) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        let seen: Seen = Default::default();
        let keep = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let mut stream = stream.unwrap();
                let mut r = BufReader::new(stream.try_clone().unwrap());
                let (mut head, mut size) = (String::new(), 0);
                loop {
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    if line.trim().is_empty() {
                        break;
                    }
                    if let Some(n) = line.to_lowercase().strip_prefix("content-length:") {
                        size = n.trim().parse().unwrap();
                    }
                    head.push_str(&line);
                }
                let mut body = vec![0; size];
                r.read_exact(&mut body).unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap();
                let content = reply(&body).to_string();
                keep.lock().unwrap().push((head, body));
                let answer =
                    serde_json::json!({"choices": [{"message": {"content": content}}]}).to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{answer}",
                    answer.len()
                )
                .unwrap();
            }
        });
        (url, seen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_remote_question_goes_as_the_openai_chat_api_and_is_kept() {
        let (url, seen) = fake::server(|_| json!({"answers": []}));
        // SAFETY: a variable only this test reads
        unsafe { std::env::set_var("T2A_TEST_KEY", "k1") };
        let place = Place::Remote {
            url,
            model: "m".into(),
            key_env: "T2A_TEST_KEY".into(),
        };
        let cache = std::env::temp_dir().join(format!("t2a-llm-{}.json", std::process::id()));
        let _ = std::fs::remove_file(&cache); // a leftover of an earlier run
        let schema = json!({"type": "object"});
        let mut llm = Llm::new(place, cache.clone()).unwrap();
        assert_eq!(
            llm.ask("sys", "café", &schema).unwrap(),
            json!({"answers": []})
        );
        llm.ask("sys", "café", &schema).unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "the second time from the cache");
        let (head, body) = &seen[0];
        assert!(head.to_lowercase().contains("authorization: bearer k1"));
        assert_eq!(
            body["messages"][1],
            json!({"role": "user", "content": "café"})
        );
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["model"], "m");
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        // llm.py's key and file: sha1 of json.dumps(["m", "sys", "café", {"type": "object"}])
        let kept = std::fs::read_to_string(&cache).unwrap();
        assert_eq!(
            kept,
            r#"{"ba7bdacfdb2a16eb19f9746ebdc5d8edf0e9c0e8": {"answers": []}}"#
        );
        std::fs::remove_file(cache).unwrap();
    }

    #[test]
    fn json_as_python_writes_it() {
        let v = json!({"a": ["é\"\n", 1, true, null], "b": {}});
        assert_eq!(
            py_json(&v, true),
            r#"{"a": ["\u00e9\"\n", 1, true, null], "b": {}}"#
        );
        assert_eq!(py_json(&json!(["😀"]), true), r#"["\ud83d\ude00"]"#);
        assert_eq!(py_json(&json!(["é"]), false), r#"["é"]"#);
        assert_eq!(py_json(&json!(["a\u{7f}b"]), true), r#"["a\u007fb"]"#);
    }

    #[test]
    fn an_unreadable_cache_or_config_is_an_error_not_an_empty_one() {
        let dir = std::env::temp_dir().join("t2a-llm-unreadable");
        std::fs::create_dir_all(dir.join("llm.json")).unwrap();
        let local = || Place::Local {
            server: "x".into(),
            gguf: "m.gguf".into(),
        };
        assert!(Llm::new(local(), dir.clone()).is_err()); // a folder: no file to read
        assert!(Llm::new(local(), dir.join("none.json")).is_ok()); // missing: nothing kept yet
        assert!(place(&dir, &BTreeMap::new()).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
