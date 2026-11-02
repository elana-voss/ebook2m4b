//! Kokoro's ONNX export patched for the GPU
//! (setup runs it once after the download; the result is
//! models/kokoro/model-gpu.onnx in the home folder):
//! - the generator's STFT (no GPU kernel in ONNX Runtime, so it ran on the CPU) becomes a Conv whose filters are
//!   the window times cos and -sin of each frequency; the DC and Nyquist bins' imaginary parts +0 exactly;
//! - the export's atan2 takes an imaginary +0 with a negative real part as pi, as torch does (Greater becomes
//!   GreaterOrEqual), and its 0/0 in a silent frame 0, not NaN (which made a whole chunk NaN; Rust only);
//! - the inverse STFT's NonZero/ScatterND division by the window sum becomes a Where over the same mask;
//! - on another card (DirectML) also: the three upsampling
//!   ConvTranspose nodes get pads 1/0 and no output_padding (same output, which DirectML otherwise refuses).
//! Kerstin's model has no STFT: it runs as it comes on CUDA, and with only the pads' change on DirectML (pads_only).
//!
//! Nodes and initializers no graph output needs any more are dropped.
//! The protobuf is read and written by a small reader here (no ONNX library): a message is its fields in order,
//! what is not touched is written back as it was. Trap: the whole model (310 MB) is held in memory about
//! three times while it is patched.
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The patch's version, in the patched model's file name: raised whenever the patch changes what it writes.
pub const VERSION: u32 = 2;
const G: &str = "/decoder/decoder/generator/";
const FLOAT: u64 = 1;
const INT64: u64 = 7;

#[derive(Clone, Debug, PartialEq)]
enum V {
    Int(u64),
    Bytes(Vec<u8>),
    F32([u8; 4]),
    F64([u8; 8]),
}

/// A protobuf message: its fields (number, value) in the order read.
type Msg = Vec<(u32, V)>;

fn varint(b: &[u8], at: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let c = *b.get(*at).context("protobuf cut short")?;
        *at += 1;
        v |= u64::from(c & 0x7f) << shift;
        if c < 0x80 {
            return Ok(v);
        }
    }
    bail!("protobuf: varint too long")
}

fn take<'a>(b: &'a [u8], at: &mut usize, n: usize) -> Result<&'a [u8]> {
    let end = at.checked_add(n).context("protobuf: length")?;
    let s = b.get(*at..end).context("protobuf cut short")?;
    *at = end;
    Ok(s)
}

fn parse(b: &[u8]) -> Result<Msg> {
    let (mut at, mut out) = (0, Vec::new());
    while at < b.len() {
        let key = varint(b, &mut at)?;
        let v = match key & 7 {
            0 => V::Int(varint(b, &mut at)?),
            1 => V::F64(take(b, &mut at, 8)?.try_into()?),
            2 => {
                let n = usize::try_from(varint(b, &mut at)?)?;
                V::Bytes(take(b, &mut at, n)?.to_vec())
            }
            5 => V::F32(take(b, &mut at, 4)?.try_into()?),
            w => bail!("protobuf: wire type {w}"),
        };
        out.push((u32::try_from(key >> 3)?, v));
    }
    Ok(out)
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn encode(m: &Msg) -> Vec<u8> {
    let mut out = Vec::new();
    for (num, v) in m {
        let wire = match v {
            V::Int(_) => 0,
            V::F64(_) => 1,
            V::Bytes(_) => 2,
            V::F32(_) => 5,
        };
        put_varint(&mut out, u64::from(*num) << 3 | wire);
        match v {
            V::Int(i) => put_varint(&mut out, *i),
            V::F64(b) => out.extend_from_slice(b),
            V::F32(b) => out.extend_from_slice(b),
            V::Bytes(b) => {
                put_varint(&mut out, b.len() as u64);
                out.extend_from_slice(b);
            }
        }
    }
    out
}

fn strs(m: &Msg, num: u32) -> Vec<String> {
    m.iter()
        .filter_map(|(n, v)| match v {
            V::Bytes(b) if *n == num => Some(String::from_utf8_lossy(b).into_owned()),
            _ => None,
        })
        .collect()
}

fn str1(m: &Msg, num: u32) -> String {
    strs(m, num).into_iter().next().unwrap_or_default()
}

fn s(text: &str) -> V {
    V::Bytes(text.as_bytes().to_vec())
}

// NodeProto: input 1, output 2, name 3, op_type 4, attribute 5. AttributeProto: name 1, i 3, ints 8, type 20.
fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: Vec<Msg>) -> Msg {
    let mut m: Msg = inputs.iter().map(|i| (1, s(i))).collect();
    m.extend(outputs.iter().map(|o| (2, s(o))));
    m.push((4, s(op)));
    m.extend(attrs.iter().map(|a| (5, V::Bytes(encode(a)))));
    m
}

fn ints(name: &str, v: &[i64]) -> Msg {
    let mut m = vec![(1, s(name))];
    m.extend(v.iter().map(|&i| (8, V::Int(i as u64))));
    m.push((20, V::Int(7))); // INTS
    m
}

fn int(name: &str, v: i64) -> Msg {
    vec![(1, s(name)), (3, V::Int(v as u64)), (20, V::Int(2))] // INT
}

/// An attribute's ints (packed or not).
fn attr_ints(a: &Msg) -> Result<Vec<i64>> {
    let mut out = Vec::new();
    for (n, v) in a {
        match (n, v) {
            (8, V::Int(i)) => out.push(*i as i64),
            (8, V::Bytes(b)) => {
                let mut at = 0;
                while at < b.len() {
                    out.push(varint(b, &mut at)? as i64);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

// TensorProto: dims 1, data_type 2, float_data 4, int64_data 7, name 8, raw_data 9.
fn tensor_f32(name: &str, dims: &[i64], v: &[f32]) -> Msg {
    let mut m: Msg = dims.iter().map(|&d| (1, V::Int(d as u64))).collect();
    m.push((2, V::Int(FLOAT)));
    m.push((8, s(name)));
    m.push((
        9,
        V::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect()),
    ));
    m
}

fn tensor_i64(name: &str, v: &[i64]) -> Msg {
    vec![
        (1, V::Int(v.len() as u64)),
        (2, V::Int(INT64)),
        (8, s(name)),
        (
            9,
            V::Bytes(v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        ),
    ]
}

/// A float or int64 tensor's values.
fn values(t: &Msg) -> Result<Vec<f64>> {
    let kind = t.iter().find_map(|(n, v)| match (n, v) {
        (2, V::Int(k)) => Some(*k),
        _ => None,
    });
    let mut out = Vec::new();
    for (n, v) in t {
        match (kind, n, v) {
            (Some(FLOAT), 9, V::Bytes(b)) => out.extend(
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f64::from(f32::from_le_bytes(*c))),
            ),
            (Some(INT64), 9, V::Bytes(b)) => out.extend(
                b.as_chunks::<8>()
                    .0
                    .iter()
                    .map(|c| i64::from_le_bytes(*c) as f64),
            ),
            (Some(FLOAT), 4, V::F32(b)) => out.push(f64::from(f32::from_le_bytes(*b))),
            (Some(FLOAT), 4, V::Bytes(b)) => out.extend(
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f64::from(f32::from_le_bytes(*c))),
            ),
            (Some(INT64), 7, V::Int(i)) => out.push(*i as i64 as f64),
            (Some(INT64), 7, V::Bytes(b)) => {
                let mut at = 0;
                while at < b.len() {
                    out.push(varint(b, &mut at)? as i64 as f64);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// STFT's filters as a Conv's weight [2 * bins, 1, n_fft]: the window times cos, then times -sin, of each bin.
fn stft_filters(window: &[f64]) -> Vec<f32> {
    let n_fft = window.len();
    let bins = n_fft / 2 + 1;
    let mut out = Vec::with_capacity(2 * bins * n_fft);
    for part in 0..2 {
        for k in 0..bins {
            for (n, w) in window.iter().enumerate() {
                let a = 2.0 * std::f64::consts::PI * (k * n) as f64 / n_fft as f64;
                let v = match part {
                    0 => w * a.cos(),
                    // +0 exactly at DC and Nyquist, as torch.stft: their phase is 0 or pi, never noise
                    _ if k == 0 || k == bins - 1 => 0.0,
                    _ => -w * a.sin(),
                };
                out.push(v as f32);
            }
        }
    }
    out
}

struct Graph {
    nodes: Vec<Msg>,
    inits: Vec<Msg>,
    /// the graph's other fields, written back as they were
    rest: Msg,
}

impl Graph {
    fn read(b: &[u8]) -> Result<Graph> {
        let mut g = Graph {
            nodes: vec![],
            inits: vec![],
            rest: vec![],
        };
        for (n, v) in parse(b)? {
            match (n, v) {
                (1, V::Bytes(b)) => g.nodes.push(parse(&b)?),
                (5, V::Bytes(b)) => g.inits.push(parse(&b)?),
                (n, v) => g.rest.push((n, v)),
            }
        }
        Ok(g)
    }

    fn write(self) -> Vec<u8> {
        let mut m: Msg = self
            .nodes
            .iter()
            .map(|n| (1, V::Bytes(encode(n))))
            .collect();
        m.extend(self.rest);
        m.extend(self.inits.iter().map(|t| (5, V::Bytes(encode(t)))));
        encode(&m)
    }

    fn at(&self, name: &str) -> Result<usize> {
        self.nodes
            .iter()
            .position(|n| str1(n, 3) == name)
            .with_context(|| format!("the model has no node {name}"))
    }

    fn init(&self, name: &str) -> Result<Vec<f64>> {
        values(
            self.inits
                .iter()
                .find(|t| str1(t, 8) == name)
                .with_context(|| format!("the model has no initializer {name}"))?,
        )
    }

    fn splice(&mut self, at: usize, new: Vec<Msg>) {
        self.nodes.splice(at..=at, new);
    }

    /// STFT as Conv, atan2 as torch's, ScatterND as Where.
    fn gpufix(&mut self) -> Result<()> {
        self.stft_as_conv()?;
        self.atan2_as_torch()?;
        self.scatter_as_where()
    }

    /// The STFT as a Conv whose filters are the window times cos and -sin of each frequency.
    fn stft_as_conv(&mut self) -> Result<()> {
        let at = self.at(&format!("{G}STFT"))?;
        let stft = &self.nodes[at];
        let (inp, out) = (strs(stft, 1), strs(stft, 2));
        let [signal, hop, window, length] = inp.as_slice() else {
            bail!("the STFT node does not take 4 inputs")
        };
        let hop = *self.init(hop)?.first().context("STFT's hop")? as i64;
        let window = self.init(window)?;
        if self.init(length)?.first() != Some(&(window.len() as f64)) {
            bail!("STFT's frame length is not its window's");
        }
        let n_fft = window.len();
        let bins = (n_fft / 2 + 1) as i64;
        let p = |x: &str| format!("{G}stft_conv.{x}");
        self.inits.extend([
            tensor_f32(
                &p("weight"),
                &[2 * bins, 1, n_fft as i64],
                &stft_filters(&window),
            ),
            tensor_f32(&p("bias"), &[2 * bins], &vec![0.0; 2 * bins as usize]), // turns -0 into +0
            tensor_i64(&p("shape"), &[0, 2, bins, -1]),
            tensor_i64(&p("in_shape"), &[0, 1, -1]),
        ]);
        let (input, conv, parts) = (p("in"), p("out"), p("parts"));
        let filters = [input.as_str(), &p("weight"), &p("bias")];
        let new = vec![
            // the signal is [batch, samples]; the Conv takes [batch, 1, samples]
            node("Reshape", &[signal, &p("in_shape")], &[&input], vec![]),
            node("Conv", &filters, &[&conv], vec![ints("strides", &[hop])]),
            node("Reshape", &[&conv, &p("shape")], &[&parts], vec![]),
            node(
                "Transpose",
                &[&parts],
                &[&out[0]],
                vec![ints("perm", &[0, 3, 2, 1])],
            ),
        ];
        self.splice(at, new);
        Ok(())
    }

    /// atan2 as torch's: an imaginary +0 with a negative real part is pi, and 0/0 is 0.
    fn atan2_as_torch(&mut self) -> Result<()> {
        // the export's atan2 adds pi to a negative real part's angle only where the imaginary part is > 0
        let at = self.at(&format!("{G}Greater"))?;
        if str1(&self.nodes[at], 1) != format!("{G}Gather_5_output_0") {
            bail!("the model's atan2 is not the one expected");
        }
        for (n, v) in self.nodes[at].iter_mut() {
            if *n == 4 {
                *v = s("GreaterOrEqual");
            }
        }
        // its imaginary over real part is 0/0 = NaN in a silent frame, which made a whole chunk NaN; torch's
        // atan2(0, 0) is 0, so a NaN quotient (its inputs are finite) becomes 0
        let at = self.at(&format!("{G}Div"))?;
        let quotient = str1(&self.nodes[at], 2);
        let a = |x: &str| format!("{G}atan2_guard.{x}");
        let (raw, nan, zero) = (a("raw"), a("nan"), a("zero"));
        for (n, v) in self.nodes[at].iter_mut() {
            if *n == 2 {
                *v = s(&raw);
            }
        }
        self.inits.push(tensor_f32(&zero, &[], &[0.0]));
        self.nodes.splice(
            at + 1..at + 1,
            [
                node("IsNaN", &[&raw], &[&nan], vec![]),
                node("Where", &[&nan, &zero, &raw], &[&quotient], vec![]),
            ],
        );
        Ok(())
    }

    /// The inverse STFT's NonZero/ScatterND division by the window sum as a Where over the same mask.
    fn scatter_as_where(&mut self) -> Result<()> {
        let i = format!("{G}istft/stft/");
        let at = self.at(&format!("{i}ScatterND"))?;
        let done = str1(&self.nodes[at], 2);
        let q = |x: &str| format!("{i}{x}");
        let (wave, mask, wsum) = (
            q("Squeeze_1_output_0"),
            q("Greater_output_0"),
            q("Slice_1_output_0"),
        );
        self.inits.extend([
            tensor_i64(&q("zero"), &[0]),
            tensor_i64(&q("last"), &[-1]),
            tensor_i64(&q("end"), &[1 << 62]),
        ]);
        let (held, head, tail, divided, head_out) =
            (q("held"), q("head"), q("tail"), q("divided"), q("head_out"));
        // the window sum is shorter than the wave: only the wave's first samples, as many as it holds, are divided
        let new = vec![
            node("Shape", &[&wsum], &[&held], vec![]),
            node(
                "Slice",
                &[&wave, &q("zero"), &held, &q("last")],
                &[&head],
                vec![],
            ),
            node(
                "Slice",
                &[&wave, &held, &q("end"), &q("last")],
                &[&tail],
                vec![],
            ),
            node("Div", &[&head, &wsum], &[&divided], vec![]),
            node("Where", &[&mask, &divided, &head], &[&head_out], vec![]),
            node(
                "Concat",
                &[&head_out, &tail],
                &[&done],
                vec![int("axis", -1)],
            ),
        ];
        self.splice(at, new);
        Ok(())
    }

    /// The upsampling ConvTranspose nodes (pads 1/1, output_padding 1) as pads 1/0.
    fn padfix(&mut self) -> Result<usize> {
        let mut fixed = 0;
        for n in self
            .nodes
            .iter_mut()
            .filter(|n| str1(n, 4) == "ConvTranspose")
        {
            let attrs: Vec<Msg> = n
                .iter()
                .filter_map(|(k, v)| match v {
                    V::Bytes(b) if *k == 5 => parse(b).ok(),
                    _ => None,
                })
                .collect();
            let get = |name: &str| attrs.iter().find(|a| str1(a, 1) == name);
            let (Some(pads), Some(extra)) = (get("pads"), get("output_padding")) else {
                continue;
            };
            if attr_ints(pads)? != [1, 1] || attr_ints(extra)? != [1] {
                continue;
            }
            n.retain(|(k, v)| {
                !(*k == 5 && matches!(v, V::Bytes(b) if parse(b).is_ok_and(|a| matches!(str1(&a, 1).as_str(), "pads" | "output_padding"))))
            });
            n.push((5, V::Bytes(encode(&ints("pads", &[1, 0])))));
            fixed += 1;
        }
        Ok(fixed)
    }

    /// Drops the nodes and initializers no graph output needs.
    fn prune(&mut self) -> Result<()> {
        let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (n, v) in &self.rest {
            if let (12, V::Bytes(b)) = (n, v) {
                needed.insert(str1(&parse(b)?, 1));
            }
        }
        let mut keep = Vec::new();
        for n in std::mem::take(&mut self.nodes).into_iter().rev() {
            if strs(&n, 2).iter().any(|o| needed.contains(o)) {
                needed.extend(strs(&n, 1));
                keep.push(n);
            }
        }
        keep.reverse();
        self.nodes = keep;
        self.inits.retain(|t| needed.contains(&str1(t, 8)));
        Ok(())
    }
}

/// Writes `src` patched for the GPU to `dst` (through a .part file); `directml` adds the pads' change.
pub fn patch(src: &Path, dst: &Path, directml: bool) -> Result<()> {
    rewrite(src, dst, |graph| {
        graph.gpufix()?;
        if directml && graph.padfix()? != 3 {
            bail!("the model's upsampling is not the one expected");
        }
        Ok(())
    })
}

/// Writes Kerstin's model to `dst` with only the pads' change, for DirectML: it has no STFT to patch.
pub fn pads_only(src: &Path, dst: &Path) -> Result<()> {
    rewrite(src, dst, |graph| {
        if graph.padfix()? == 0 {
            bail!("the model's upsampling is not the one expected");
        }
        Ok(())
    })
}

fn rewrite(src: &Path, dst: &Path, change: impl FnOnce(&mut Graph) -> Result<()>) -> Result<()> {
    let bytes = std::fs::read(src).with_context(|| src.display().to_string())?;
    let mut model = parse(&bytes)?;
    drop(bytes);
    let at = model
        .iter()
        .position(|(n, _)| *n == 7)
        .context("the model has no graph")?;
    let V::Bytes(g) = &model[at].1 else {
        bail!("the model's graph is no message")
    };
    let mut graph = Graph::read(g)?;
    change(&mut graph)?;
    graph.prune()?;
    model[at].1 = V::Bytes(graph.write());
    let part = dst.with_extension("part");
    std::fs::create_dir_all(dst.parent().context("no folder")?)?;
    std::fs::write(&part, encode(&model))?;
    std::fs::rename(&part, dst)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_is_written_back_as_it_was_read() {
        let m: Msg = vec![
            (1, V::Int(300)),
            (2, s("name")),
            (4, V::F32([1, 2, 3, 4])),
            (5, V::F64([0; 8])),
        ];
        let b = encode(&m);
        assert_eq!(parse(&b).unwrap(), m);
        assert!(parse(&b[..b.len() - 1]).is_err());
    }

    #[test]
    fn the_conv_filters_are_the_stft() {
        let n = 20;
        let window: Vec<f64> = (0..n)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n as f64).cos())
            .collect();
        let f = stft_filters(&window);
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 % 11) as f64 - 5.0) / 3.0).collect();
        let bins = n / 2 + 1;
        for k in 0..bins {
            let (mut re, mut im) = (0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let a = 2.0 * std::f64::consts::PI * (k * i) as f64 / n as f64;
                re += v * window[i] * a.cos();
                im -= v * window[i] * a.sin();
            }
            let dot = |row: usize| {
                (0..n)
                    .map(|i| f64::from(f[row * n + i]) * x[i])
                    .sum::<f64>()
            };
            assert!((dot(k) - re).abs() < 1e-4 && (dot(bins + k) - im).abs() < 1e-4);
        }
        assert!(f[bins * n..(bins + 1) * n].iter().all(|v| v.to_bits() == 0)); // DC's imaginary +0
    }

    fn tiny() -> Graph {
        let i = format!("{G}istft/stft/");
        let out = |n: &str| vec![(1, s(n))];
        Graph {
            nodes: vec![
                node("Identity", &["x"], &["sig"], vec![]),
                [
                    vec![(3, s(&format!("{G}STFT")))],
                    node("STFT", &["sig", "hop", "win", "len"], &["spec"], vec![]),
                ]
                .concat(),
                [
                    vec![(3, s(&format!("{G}Greater")))],
                    node(
                        "Greater",
                        &[&format!("{G}Gather_5_output_0"), "z"],
                        &["g"],
                        vec![],
                    ),
                ]
                .concat(),
                [
                    vec![(3, s(&format!("{G}Div")))],
                    node(
                        "Div",
                        &[
                            &format!("{G}Gather_5_output_0"),
                            &format!("{G}Gather_4_output_0"),
                        ],
                        &[&format!("{G}Div_output_0")],
                        vec![],
                    ),
                ]
                .concat(),
                node("Atan", &[&format!("{G}Div_output_0")], &["angle"], vec![]),
                node("NonZero", &["spec"], &["unused"], vec![]),
                [
                    vec![(3, s(&format!("{i}ScatterND")))],
                    node("ScatterND", &["spec", "g"], &["y"], vec![]),
                ]
                .concat(),
                node(
                    "ConvTranspose",
                    &["y", "g", "spec", "angle"], // the STFT and the atan2 are needed further on, as in Kokoro
                    &["up"],
                    vec![ints("pads", &[1, 1]), ints("output_padding", &[1])],
                ),
            ],
            inits: vec![
                tensor_i64("hop", &[5]),
                tensor_f32("win", &[4], &[0.0, 0.5, 1.0, 0.5]),
                tensor_i64("len", &[4]),
                tensor_i64("orphan", &[1]),
            ],
            rest: vec![(12, V::Bytes(encode(&out("up"))))],
        }
    }

    #[test]
    fn the_atan2_of_a_silent_frame_is_0_not_nan() {
        let mut g = Graph::read(&tiny().write()).unwrap();
        g.gpufix().unwrap();
        g.prune().unwrap();
        let by_op = |op: &str| g.nodes.iter().find(|n| str1(n, 4) == op).unwrap().clone();
        let div = g.nodes[g.at(&format!("{G}Div")).unwrap()].clone();
        let (isnan, atan) = (by_op("IsNaN"), by_op("Atan"));
        assert_eq!(strs(&isnan, 1), strs(&div, 2)); // 0/0 is found
        let guard = g
            .nodes
            .iter()
            .find(|n| str1(n, 4) == "Where" && strs(n, 2) == strs(&atan, 1))
            .expect("Atan reads a Where");
        let ins = strs(guard, 1);
        assert_eq!((&ins[0], &ins[2]), (&strs(&isnan, 2)[0], &strs(&div, 2)[0]));
        assert_eq!(g.init(&ins[1]).unwrap(), [0.0]); // torch's atan2(0, 0)
    }

    #[test]
    fn the_patch_rewrites_stft_atan2_and_scatter_and_drops_what_is_unused() {
        let mut g = Graph::read(&tiny().write()).unwrap();
        g.gpufix().unwrap();
        assert_eq!(g.padfix().unwrap(), 1);
        g.prune().unwrap();
        let ops: Vec<String> = g.nodes.iter().map(|n| str1(n, 4)).collect();
        assert!(
            !ops.iter()
                .any(|o| ["STFT", "ScatterND", "NonZero", "Greater"].contains(&o.as_str()))
        );
        assert!(
            ops.contains(&"Conv".into())
                && ops.contains(&"Where".into())
                && ops.contains(&"GreaterOrEqual".into()),
            "{ops:?}"
        );
        let conv = g.nodes.iter().find(|n| str1(n, 4) == "Conv").unwrap();
        let strides = parse(match &conv.iter().find(|(k, _)| *k == 5).unwrap().1 {
            V::Bytes(b) => b,
            _ => panic!(),
        })
        .unwrap();
        assert_eq!(attr_ints(&strides).unwrap(), [5]);
        let weight = g.init(&format!("{G}stft_conv.weight")).unwrap();
        assert_eq!(weight.len(), 2 * 3 * 4);
        assert!(g.init("orphan").is_err() && g.init("hop").is_err()); // nothing reads them any more
        let up = g.nodes.last().unwrap();
        let pads: Vec<Vec<i64>> = up
            .iter()
            .filter_map(|(k, v)| match v {
                V::Bytes(b) if *k == 5 => attr_ints(&parse(b).unwrap()).ok(),
                _ => None,
            })
            .collect();
        assert_eq!(pads, [vec![1, 0]]);
    }

    #[test]
    fn pads_only_changes_the_pads_and_leaves_the_rest_of_the_graph() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-pads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (src, dst) = (dir.join("in.onnx"), dir.join("out.onnx"));
        std::fs::write(&src, encode(&vec![(7, V::Bytes(tiny().write()))])).unwrap();
        pads_only(&src, &dst).unwrap();
        let model = parse(&std::fs::read(&dst).unwrap()).unwrap();
        let V::Bytes(g) = &model[0].1 else { panic!() };
        let g = Graph::read(g).unwrap();
        let ops: Vec<String> = g.nodes.iter().map(|n| str1(n, 4)).collect();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(ops.contains(&"STFT".into()) && !ops.contains(&"Conv".into()), "{ops:?}");
        let up = g.nodes.iter().find(|n| str1(n, 4) == "ConvTranspose").unwrap();
        let attrs: Vec<(String, Vec<i64>)> = up
            .iter()
            .filter_map(|(k, v)| match v {
                V::Bytes(b) if *k == 5 => parse(b).ok().map(|a| (str1(&a, 1), attr_ints(&a).unwrap())),
                _ => None,
            })
            .collect();
        assert_eq!(attrs, [("pads".to_string(), vec![1, 0])]);
    }
}
