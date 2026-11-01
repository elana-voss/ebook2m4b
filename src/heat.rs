//! The heat wait (the hardware's safety): before each chunk and each question to the language model
//! the GPU's and CPU's temperatures are looked at, at most every 30 s; at GPU 83 C or CPU 90 C the run
//! waits, looking again every 30 s, until both are below. The GPU's from nvidia-smi (none on another
//! card: 0), the CPU's from Windows' thermal zones (tenths of kelvin, the hottest zone).
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const HOT_GPU: f32 = 83.0;
const HOT_CPU: f32 = 90.0;
const EVERY: Duration = Duration::from_secs(30);

/// GPU and CPU in degrees Celsius; 0 where nothing can be read.
fn temperatures() -> (f32, f32) {
    let run = |exe: &str, args: &[&str]| {
        Command::new(exe)
            .args(args)
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            .unwrap_or_default() // no reader: taken as cool, as t2a.py does
    };
    let gpu = run(
        "nvidia-smi",
        &["--query-gpu=temperature.gpu", "--format=csv,noheader"],
    );
    let cpu = run(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | Measure-Object HighPrecisionTemperature -Maximum).Maximum",
        ],
    );
    parse(&gpu, &cpu)
}

fn parse(gpu: &str, cpu: &str) -> (f32, f32) {
    let gpu = gpu
        .split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or(0.0);
    let cpu = cpu.trim().parse::<f32>().map_or(0.0, |t| t / 10.0 - 273.15);
    (gpu, cpu)
}

fn hot((gpu, cpu): (f32, f32)) -> bool {
    gpu >= HOT_GPU || cpu >= HOT_CPU
}

/// Waits while the machine is hot; looks at most every 30 s, shared by the whole run.
pub fn wait() {
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    if last.is_some_and(|t| t.elapsed() < EVERY) {
        return;
    }
    loop {
        let t = temperatures();
        if !hot(t) {
            break;
        }
        println!("  hot (GPU {:.0} C, CPU {:.0} C): waiting", t.0, t.1);
        std::thread::sleep(EVERY);
    }
    *last = Some(Instant::now());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperatures_are_read_and_judged() {
        let t = parse("61\n", "3631\r\n");
        assert!((t.0 - 61.0).abs() < 0.01 && (t.1 - 89.95).abs() < 0.01 && !hot(t));
        assert!(hot(parse("83", "")) && hot(parse("", "3632")) && !hot(parse("", "")));
    }
}
