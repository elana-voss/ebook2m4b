//! Linux: Whisper on the NVIDIA card for ebook2m4b, which starts this program beside it (whisper.rs): the
//! model (its one argument) loaded once on CUDA (ear.rs), then each question on stdin (the hint and Kokoro's
//! 24 kHz samples, frames.rs) answered on stdout with what it heard, until stdin closes. whisper.cpp's own
//! messages go to stderr. A separate program so that ebook2m4b itself starts without CUDA's libraries, which
//! setup fetches; this one is started after setup, with them on LD_LIBRARY_PATH.
//! On Windows it only says so: ebook2m4b runs Whisper itself there.
#[cfg(target_os = "linux")]
#[path = "../ear.rs"]
mod ear;
#[cfg(target_os = "linux")]
#[path = "../frames.rs"]
mod frames;

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    let model = std::env::args_os()
        .nth(1)
        .context("usage: ebook2m4b-whisper <Whisper model>")?;
    let mut state = ear::built(std::path::Path::new(&model))?;
    let (mut input, mut output) = (std::io::stdin().lock(), std::io::stdout().lock());
    while let Some((hint, wav)) = frames::question(&mut input)? {
        let text = ear::hear(&mut state, &wav, 24_000, &hint)?;
        frames::answer(&mut output, &text)?;
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("ebook2m4b-whisper is Linux's: on Windows ebook2m4b runs Whisper itself");
    std::process::exit(2);
}
