//! Hex digests, the one module naming sha2: of setup's downloads, and a key of JSON (llm.rs's cache,
//! speak.rs's work id).

/// The SHA-256 of `v` as compact JSON.
pub fn json_key(v: &serde_json::Value) -> String {
    format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(v.to_string()))
}

/// SHA-256 of what is written to it.
#[derive(Default)]
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn hex(self) -> String {
        format!("{:x}", sha2::Digest::finalize(self.0))
    }
}

impl std::io::Write for Sha256 {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        sha2::Digest::update(&mut self.0, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
