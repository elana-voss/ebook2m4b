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
//! The heat wait (the hardware's safety): one thread reads the GPU's and CPU's temperatures every 30 s (the
//! readers take seconds, so the speech does not wait for them); before each chunk and each question to the
//! language model the latest reading is looked at, the first call waiting for the first reading; at GPU 83 C or CPU 90 C the run
//! waits until a reading has both below. The GPU's from nvidia-smi (none on another
//! card: 0), the CPU's from the hottest thermal zone: Windows' (tenths of kelvin, by PowerShell), Linux's
//! /sys/class/thermal/thermal_zone*/temp and its CPU drivers' /sys/class/hwmon/hwmon*/temp*_input
//! (thousandths of a degree; WSL2 has none: 0).
//!
//! in_order(): a job's items on up to CORES threads, the results in the items' order.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Os {
    Windows,
    Linux,
}

pub const THIS: Os = if cfg!(windows) { Os::Windows } else { Os::Linux };

/// The threads a heavy job takes at most: half the physical cores of the machine it is built on.
pub const CORES: usize = 4;

/// `work` of each item (its index, the item) on up to CORES threads, the results in the items' order;
/// a panic in one is raised here.
pub fn in_order<T: Sync, R: Send>(items: &[T], work: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    let next = AtomicUsize::new(0);
    let mut done: Vec<(usize, R)> = std::thread::scope(|s| {
        let turn = || {
            let mut got = Vec::new();
            loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = items.get(i) else { return got };
                got.push((i, work(i, item)));
            }
        };
        let threads: Vec<_> = (0..CORES.min(items.len())).map(|_| s.spawn(turn)).collect();
        threads
            .into_iter()
            .flat_map(|t| t.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });
    done.sort_by_key(|d| d.0);
    done.into_iter().map(|d| d.1).collect()
}

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

/// The latest reading, made on a thread of its own so that a wait costs the speech nothing.
struct Watch {
    latest: Mutex<Option<(f32, f32)>>,
    read: Condvar,
}

impl Watch {
    /// Starts the thread: `read` at once and then every `every`. It ends with the program.
    fn start(read: impl Fn() -> (f32, f32) + Send + 'static, every: Duration) -> Arc<Watch> {
        let w = Arc::new(Watch { latest: Mutex::new(None), read: Condvar::new() });
        let own = Arc::clone(&w);
        std::thread::spawn(move || {
            loop {
                let t = read(); // no lock held across the reading
                *own.latest.lock().unwrap_or_else(|e| e.into_inner()) = Some(t);
                own.read.notify_all();
                std::thread::sleep(every);
            }
        });
        w
    }

    /// The latest reading; before the first one it waits for it, so nothing starts hot.
    fn now(&self) -> (f32, f32) {
        let mut t = self.latest.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(r) = *t {
                return r;
            }
            t = self.read.wait(t).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Sleeps `nap` at a time while the latest reading says hot.
    fn wait_while_hot(&self, nap: Duration) {
        loop {
            let t = self.now();
            if !hot(t) {
                return;
            }
            println!("  hot (GPU {:.0} C, CPU {:.0} C): waiting", t.0, t.1);
            std::thread::sleep(nap);
        }
    }
}

/// Waits while the machine is hot, by the one watch of the whole run (started by the first call).
pub fn wait_while_hot() {
    static WATCH: LazyLock<Arc<Watch>> = LazyLock::new(|| Watch::start(temperatures, EVERY));
    WATCH.wait_while_hot(EVERY);
}

#[cfg(test)]
mod tests {
    use super::in_order;

    #[test]
    fn in_order_keeps_the_items_order(){
        let items: Vec<usize> = (0..1000).collect();
        let got = in_order(&items, |i, x| i + *x);
        assert_eq!(got, (0..1000).map(|x| 2 * x).collect::<Vec<_>>());
        assert!(in_order(&[] as &[usize], |_, x| *x).is_empty());
    }

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
    fn the_wait_never_waits_for_a_reader_after_the_first_reading() {
        let reads = Arc::new(Mutex::new(0));
        let n = Arc::clone(&reads);
        let w = Watch::start(
            move || {
                *n.lock().unwrap() += 1;
                std::thread::sleep(Duration::from_millis(300));
                (40.0, 40.0)
            },
            Duration::from_secs(60),
        );
        w.wait_while_hot(Duration::ZERO); // the first reading: 300 ms
        let t = std::time::Instant::now();
        for _ in 0..100 {
            w.wait_while_hot(Duration::ZERO);
        }
        assert!(t.elapsed() < Duration::from_millis(100), "{:?}", t.elapsed());
        assert_eq!(*reads.lock().unwrap(), 1);
    }

    #[test]
    fn a_hot_reading_holds_the_wait_until_a_cool_one() {
        let reads = Arc::new(Mutex::new(0));
        let n = Arc::clone(&reads);
        let w = Watch::start(
            move || {
                let mut k = n.lock().unwrap();
                *k += 1;
                if *k < 3 { (HOT_GPU, 0.0) } else { (0.0, 0.0) }
            },
            Duration::from_millis(10),
        );
        w.wait_while_hot(Duration::from_millis(1));
        assert!(*reads.lock().unwrap() >= 3);
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
