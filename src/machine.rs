//! This machine: its system's names, its card and its heat.
//!
//! The two systems the program is built for, Windows and Linux (macOS not), and what differs between them
//! by name alone: the home folder, a program's file name, ONNX Runtime's library. Each is a function of the
//! system, so both are tested on either; THIS is the one built for.
//!
//! Which graphics card runs the models: an NVIDIA card (CUDA) or another one (DirectML and Vulkan).
//! NVIDIA's driver brings nvidia-smi (on PATH on Windows and on Linux, WSL2 included); a card it lists is taken
//! as NVIDIA. Linux has the NVIDIA route only: setup (downloads.rs) refuses another card there.
//! The hidden --card other (main.rs) takes the other route on any machine, to test it where NVIDIA is found.
//!
//! The heat wait (the hardware's safety): before each chunk and each question to the language model
//! the GPU's and CPU's temperatures are looked at, at most every 30 s; at GPU 83 C or CPU 90 C the run
//! waits, looking again every 30 s, until both are below. The GPU's from nvidia-smi (none on another
//! card: 0), the CPU's from the hottest thermal zone: Windows' (tenths of kelvin, by PowerShell), Linux's
//! /sys/class/thermal/thermal_zone*/temp and its CPU drivers' /sys/class/hwmon/hwmon*/temp*_input
//! (thousandths of a degree; WSL2 has none: 0).
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Os {
    Windows,
    Linux,
}

pub const THIS: Os = if cfg!(windows) { Os::Windows } else { Os::Linux };

/// The home folder: %LOCALAPPDATA%\ebook2m4b on Windows; on Linux $XDG_DATA_HOME/ebook2m4b, else
/// ~/.local/share/ebook2m4b. `var` reads an environment variable (one set but empty counts as unset).
pub fn home(os: Os, var: impl Fn(&str) -> Option<String>) -> Result<PathBuf> {
    let var = |k: &str| var(k).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = match os {
        Os::Windows => var("LOCALAPPDATA").context("LOCALAPPDATA is not set")?,
        Os::Linux => match var("XDG_DATA_HOME") {
            Some(d) => d,
            None => var("HOME")
                .context("neither XDG_DATA_HOME nor HOME is set")?
                .join(".local/share"),
        },
    };
    Ok(base.join("ebook2m4b"))
}

/// A program's file name: `name`.exe on Windows, `name` on Linux.
pub fn exe(os: Os, name: &str) -> String {
    match os {
        Os::Windows => format!("{name}.exe"),
        Os::Linux => name.to_string(),
    }
}

/// ONNX Runtime's library as setup unpacks it.
pub fn onnxruntime(os: Os) -> &'static str {
    match os {
        Os::Windows => "onnxruntime.dll",
        Os::Linux => "libonnxruntime.so.1",
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Card {
    Nvidia,
    Other,
}

pub fn card() -> Card {
    match Command::new("nvidia-smi").arg("-L").output() {
        Ok(out)
            if out.status.success() && String::from_utf8_lossy(&out.stdout).contains("GPU ") =>
        {
            Card::Nvidia
        }
        _ => Card::Other,
    }
}

/// The card asked for by the hidden --card ("other"), else the one found.
pub fn chosen(forced: Option<&str>) -> Card {
    match forced {
        Some("other") => Card::Other,
        _ => card(),
    }
}

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
            .unwrap_or_default() // no reader: taken as cool
    };
    let gpu = run(
        "nvidia-smi",
        &["--query-gpu=temperature.gpu", "--format=csv,noheader"],
    );
    #[cfg(windows)]
    let cpu = windows_cpu(&run(
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation | Measure-Object HighPrecisionTemperature -Maximum).Maximum",
        ],
    ));
    #[cfg(unix)]
    let cpu = linux_cpu(&linux_sensors(std::path::Path::new("/sys")));
    (gpu_c(&gpu), cpu)
}

/// nvidia-smi's first card's temperature.
fn gpu_c(out: &str) -> f32 {
    out.split_whitespace()
        .next()
        .and_then(|t| t.parse().ok())
        .unwrap_or(0.0)
}

/// Windows' hottest zone, in tenths of kelvin.
#[cfg_attr(not(windows), allow(dead_code))] // tested on both systems
fn windows_cpu(out: &str) -> f32 {
    out.trim().parse::<f32>().map_or(0.0, |t| t / 10.0 - 273.15)
}

/// The hwmon drivers of CPUs (Intel's, AMD's two).
const CPU_DRIVERS: [&str; 3] = ["coretemp", "k10temp", "zenpower"];

/// What Linux's CPU sensors under `sys` (/sys) say: every thermal zone's temp, and each temp*_input of
/// a hwmon device a CPU's driver names (a CPU without a thermal zone).
#[cfg_attr(not(unix), allow(dead_code))] // tested on both systems
fn linux_sensors(sys: &Path) -> Vec<String> {
    let entries = |dir: &Path| -> Vec<PathBuf> {
        let list = std::fs::read_dir(dir).into_iter().flatten().flatten();
        list.map(|e| e.path()).collect()
    };
    let named = |p: &Path, start: &str, end: &str| {
        let n = p.file_name().unwrap_or_default().to_string_lossy();
        n.starts_with(start) && n.ends_with(end)
    };
    let read = |p: &Path| std::fs::read_to_string(p).ok();
    let zones = entries(&sys.join("class/thermal")).into_iter();
    let mut found: Vec<String> = zones
        .filter(|p| named(p, "thermal_zone", ""))
        .filter_map(|p| read(&p.join("temp")))
        .collect();
    for dev in entries(&sys.join("class/hwmon")) {
        let name = read(&dev.join("name")).unwrap_or_default();
        if CPU_DRIVERS.contains(&name.trim()) {
            let inputs = entries(&dev).into_iter().filter(|p| named(p, "temp", "_input"));
            found.extend(inputs.filter_map(|p| read(&p)));
        }
    }
    found
}

/// Linux's sensors' readings, each in thousandths of a degree: the hottest.
#[cfg_attr(not(unix), allow(dead_code))] // tested on both systems
fn linux_cpu(zones: &[String]) -> f32 {
    zones
        .iter()
        .filter_map(|z| z.trim().parse::<f32>().ok())
        .fold(0.0, f32::max)
        / 1000.0
}

fn hot((gpu, cpu): (f32, f32)) -> bool {
    gpu >= HOT_GPU || cpu >= HOT_CPU
}

/// Waits while the machine is hot; looks at most every 30 s, shared by the whole run.
pub fn wait_while_hot() {
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

    fn env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn each_system_has_its_home_folder() {
        let w = home(Os::Windows, env(&[("LOCALAPPDATA", r"C:\L")])).unwrap();
        assert_eq!(w, PathBuf::from(r"C:\L").join("ebook2m4b"));
        let x = home(Os::Linux, env(&[("XDG_DATA_HOME", "/d"), ("HOME", "/h")])).unwrap();
        assert_eq!(x, PathBuf::from("/d").join("ebook2m4b"));
        let h = home(Os::Linux, env(&[("XDG_DATA_HOME", ""), ("HOME", "/h")])).unwrap();
        assert_eq!(h, PathBuf::from("/h").join(".local/share").join("ebook2m4b"));
        assert!(home(Os::Linux, env(&[])).is_err());
        assert!(home(Os::Windows, env(&[("HOME", "/h")])).is_err());
    }

    #[test]
    fn programs_and_libraries_are_named_per_system() {
        assert_eq!(exe(Os::Windows, "llama-server"), "llama-server.exe");
        assert_eq!(exe(Os::Linux, "llama-server"), "llama-server");
        assert_ne!(onnxruntime(Os::Windows), onnxruntime(Os::Linux));
    }

    #[test]
    fn other_asked_for_is_taken_whatever_card_is_found() {
        assert_eq!(chosen(Some("other")), Card::Other);
    }

    #[test]
    fn temperatures_are_read_and_judged() {
        let t = (gpu_c("61\n"), windows_cpu("3631\r\n"));
        assert!((t.0 - 61.0).abs() < 0.01 && (t.1 - 89.95).abs() < 0.01 && !hot(t));
        assert!(hot((gpu_c("83"), 0.0)) && hot((0.0, windows_cpu("3632"))));
        assert!(!hot((gpu_c(""), windows_cpu(""))));
        let zones = ["45000\n".to_string(), "90500\n".into(), "x".into()];
        assert!(hot((0.0, linux_cpu(&zones))) && linux_cpu(&[]) == 0.0);
    }

    #[test]
    fn a_cpu_seen_only_by_hwmon_is_read() {
        let sys = std::env::temp_dir().join(format!("ebook2m4b-heat-{}", std::process::id()));
        let put = |p: &str, s: &str| {
            let f = sys.join(p);
            std::fs::create_dir_all(f.parent().unwrap()).unwrap();
            std::fs::write(f, s).unwrap();
        };
        put("class/hwmon/hwmon0/name", "k10temp\n");
        put("class/hwmon/hwmon0/temp1_input", "91000\n");
        put("class/hwmon/hwmon1/name", "nvme\n");
        put("class/hwmon/hwmon1/temp1_input", "95000\n");
        let cpu = linux_cpu(&linux_sensors(&sys));
        std::fs::remove_dir_all(&sys).unwrap();
        assert!((cpu - 91.0).abs() < 0.01, "{cpu}");
    }
}
