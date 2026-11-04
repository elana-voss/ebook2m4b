//! The program's WAVs: mono 32-bit float at Kokoro's rate, written (kokoro's samples, speak.rs's chunks,
//! the sound sent to Whisper's server) and read back (assemble.rs).
use anyhow::{Context, Result, bail};
use std::path::Path;

/// samples a second: what Kokoro speaks, so every WAV here
pub const RATE: u32 = 24_000;

/// A 32-bit float WAV of `samples` at `rate`, one channel.
pub fn wav(samples: &[f32], rate: u32) -> Vec<u8> {
    let data = (samples.len() * 4) as u32;
    let mut b = Vec::with_capacity(44 + data as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&rate.to_le_bytes());
    b.extend_from_slice(&(rate * 4).to_le_bytes());
    b.extend_from_slice(&4u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data.to_le_bytes());
    put_samples(&mut b, samples).expect("a Vec takes all");
    b
}

/// Samples as a WAV's data: 32-bit float, little-endian.
pub fn put_samples(to: &mut impl std::io::Write, samples: &[f32]) -> std::io::Result<()> {
    for s in samples {
        to.write_all(&s.to_le_bytes())?;
    }
    Ok(())
}

/// A mono WAV's samples, 32-bit float as speak.rs writes them.
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).with_context(|| path.display().to_string())?;
    if b.len() < 12 || &b[..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        bail!("{} is no WAV", path.display());
    }
    let (mut at, mut fmt) = (12, None);
    while at + 8 <= b.len() {
        let id = &b[at..at + 4];
        let len = u32::from_le_bytes(b[at + 4..at + 8].try_into()?) as usize;
        let body = &b[at + 8..(at + 8 + len).min(b.len())];
        if id == b"fmt " && body.len() >= 16 {
            let tag = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]);
            let bits = u16::from_le_bytes([body[14], body[15]]);
            fmt = Some((tag, channels, bits));
        } else if id == b"data" {
            return match fmt {
                Some((3, 1, 32)) => Ok(body
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|s| f32::from_le_bytes(*s))
                    .collect()),
                f => bail!("{}: a WAV of format {f:?} is not read", path.display()),
            };
        }
        at += 8 + len + len % 2;
    }
    bail!("{} has no sound", path.display())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_wav_holds_its_samples_after_a_44_byte_head() {
        let b = super::wav(&[0.5, -1.0], 24_000);
        assert_eq!(b.len(), 52);
        assert_eq!(&b[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 44);
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 24_000);
        assert_eq!(f32::from_le_bytes(b[48..52].try_into().unwrap()), -1.0);
    }
}
