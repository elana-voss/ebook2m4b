//! The heat wait (the hardware's safety): before each chunk and each question to the language model
//! the GPU's and CPU's temperatures are looked at, at most every 30 s; at GPU 83 C or CPU 90 C the run
//! waits, looking again every 30 s, until both are below. The GPU's from nvidia-smi (none on another
//! card: 0), the CPU's from the hottest thermal zone: Windows' (tenths of kelvin, by PowerShell), Linux's
//! /sys/class/thermal/thermal_zone*/temp and its CPU drivers' /sys/class/hwmon/hwmon*/temp*_input
//! (thousandths of a degree; WSL2 has none: 0).
use std::path::{Path, PathBuf};
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
        let t = (gpu_c("61\n"), windows_cpu("3631\r\n"));
        assert!((t.0 - 61.0).abs() < 0.01 && (t.1 - 89.95).abs() < 0.01 && !hot(t));
        assert!(hot((gpu_c("83"), 0.0)) && hot((0.0, windows_cpu("3632"))));
        assert!(!hot((gpu_c(""), windows_cpu(""))));
        let zones = ["45000\n".to_string(), "90500\n".into(), "x".into()];
        assert!(hot((0.0, linux_cpu(&zones))) && linux_cpu(&[]) == 0.0);
    }

    #[test]
    fn a_cpu_seen_only_by_hwmon_is_read() {
        let sys = std::env::temp_dir().join(format!("t2a-heat-{}", std::process::id()));
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
