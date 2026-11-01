//! Hex digests, the one module naming sha1 and sha2: SHA-1 for the keys shared with the Python version
//! (llm.rs's cache, speak.rs's work id), SHA-256 for setup's downloads.

pub fn sha1(data: impl AsRef<[u8]>) -> String {
    format!("{:x}", <sha1::Sha1 as sha1::Digest>::digest(data))
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
