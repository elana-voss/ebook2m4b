//! Kokoro's ONNX export patched for the GPU, and both speech models' silence between words limited
//! (setup runs it once after the download; the result is models/<kokoro|kerstin>/model-gpu[-dml]-v<VERSION>.onnx
//! in the home folder):
//! - the generator's STFT (no GPU kernel in ONNX Runtime, so it ran on the CPU) becomes a Conv whose filters are
//!   the window times cos and -sin of each frequency; the DC and Nyquist bins' imaginary parts +0 exactly;
//! - the export's atan2 takes an imaginary +0 with a negative real part as pi, as torch does (Greater becomes
//!   GreaterOrEqual), and its 0/0 in a silent frame 0, not NaN (which made a whole chunk NaN; Rust only);
//! - the inverse STFT's NonZero/ScatterND division by the window sum becomes a Where over the same mask;
//! - on another card (DirectML) also: the three upsampling
//!   ConvTranspose nodes get pads 1/0 and no output_padding (same output, which DirectML otherwise refuses);
//! - both models (Kokoro's and Kerstin's): after the Clip that times each token, a word's last sound and the
//!   space after it, with no punctuation between the two words, if together over Round(L/speed) frames (Kokoro 8,
//!   Kerstin 10 at speed 1: the top of the normal range measured; above it the voice made silences of 0.2-0.44 s
//!   mid-sentence) get Round(S/speed) (Kokoro 4, Kerstin 3: such a pair renders silent over nearly all its
//!   frames, about 0.025 s each, so S frames stay under the measured top); every other token as it was
//!   (limit_word_gaps).
//!
//! Kerstin's model has no STFT: it gets the limit, and the pads' change on DirectML.
//!
//! Nodes and initializers no graph output needs any more are dropped.
//! The protobuf is read and written by a small reader here (no ONNX library): a message is its fields in order,
//! what is not touched is written back as it was. Trap: the whole model (310 MB) is held in memory about
//! three times while it is patched.
use anyhow::{Context, Result, bail};
use std::path::Path;

/// The patch's version, in the patched model's file name: raised whenever the patch changes what it writes.
pub const VERSION: u32 = 4;
const G: &str = "/decoder/decoder/generator/";
const FLOAT: u64 = 1;
const INT64: u64 = 7;
/// Kokoro's token ids: a space, and the first sound (below it the padding and punctuation).
const SPACE: i64 = 16;
const FIRST_SOUND: i64 = 17;

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

/// New nodes, each output named by its op and a count.
struct Nodes {
    new: Vec<Msg>,
    made: usize,
}

impl Nodes {
    fn op(&mut self, op: &str, inputs: &[&str], attrs: Vec<Msg>) -> String {
        self.made += 1;
        let out = format!("/word_gap/{op}_{}", self.made);
        self.new.push(node(op, inputs, &[&out], attrs));
        out
    }
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
            let padding = |b: &[u8]| {
                parse(b).is_ok_and(|a| matches!(str1(&a, 1).as_str(), "pads" | "output_padding"))
            };
            n.retain(|(k, v)| !(*k == 5 && matches!(v, V::Bytes(b) if padding(b))));
            n.push((5, V::Bytes(encode(&ints("pads", &[1, 0])))));
            fixed += 1;
        }
        Ok(fixed)
    }

    /// Limits the silence between two words with no punctuation between them: after the Clip that times each
    /// token, where token i is a space and both neighbours are sounds or marks (id 17 or more), and tokens i-1
    /// and i together have more than Round(limit / speed) frames, the two get Round(set / speed) (at least 2),
    /// token i-1 shortened first, each at least 1; a pair at or under the limit stays as it was.
    /// The new nodes write the Clip's old output, so what read it reads the limited frames.
    fn limit_word_gaps(&mut self, clip: &str, limit: i64, set: i64) -> Result<()> {
        let inputs: Vec<String> = self
            .rest
            .iter()
            .filter_map(|(n, v)| match v {
                V::Bytes(b) if *n == 11 => parse(b).ok().map(|i| str1(&i, 1)),
                _ => None,
            })
            .collect();
        if !["input_ids", "speed"].iter().all(|i| inputs.iter().any(|x| x == i)) {
            bail!("the model does not take input_ids and speed");
        }
        let at = self.at(clip)?;
        let frames = strs(&self.nodes[at], 2)
            .into_iter()
            .next()
            .context("the Clip has no output")?;
        let p = |x: &str| format!("/word_gap/{x}");
        let raw = p("unlimited");
        for (n, v) in self.nodes[at].iter_mut() {
            if *n == 2 {
                *v = s(&raw);
            }
        }
        self.inits.extend([
            tensor_i64(&p("space"), &[SPACE]),
            tensor_i64(&p("sound"), &[FIRST_SOUND]),
            tensor_f32(&p("limit"), &[1], &[limit as f32]),
            tensor_f32(&p("set"), &[1], &[set as f32]),
            tensor_f32(&p("0"), &[1], &[0.0]),
            tensor_f32(&p("1"), &[1], &[1.0]),
            tensor_f32(&p("2"), &[1], &[2.0]),
            tensor_i64(&p("i0"), &[0]),
            tensor_i64(&p("i1"), &[1]),
            tensor_i64(&p("i-1"), &[-1]),
            tensor_i64(&p("end"), &[i64::MAX]),
        ]);
        let mut b = Nodes { new: vec![], made: 0 };
        let f = |b: &mut Nodes, x: &str| b.op("Cast", &[x], vec![int("to", FLOAT as i64)]);
        let space = b.op("Equal", &["input_ids", &p("space")], vec![]);
        let space = f(&mut b, &space);
        let sound = b.op("GreaterOrEqual", &["input_ids", &p("sound")], vec![]);
        let sound = f(&mut b, &sound);
        // a column of zeros as wide as one token, to shift a row by one token either way
        let first = b.op("Slice", &[&raw, &p("i0"), &p("i1"), &p("i-1")], vec![]);
        let zero = b.op("Mul", &[&first, &p("0")], vec![]);
        let shift = |b: &mut Nodes, x: &str, from_before: bool| {
            if from_before {
                let c = b.op("Concat", &[&zero, x], vec![int("axis", -1)]);
                b.op("Slice", &[&c, &p("i0"), &p("i-1"), &p("i-1")], vec![])
            } else {
                let c = b.op("Concat", &[x, &zero], vec![int("axis", -1)]);
                b.op("Slice", &[&c, &p("i1"), &p("end"), &p("i-1")], vec![])
            }
        };
        let before = shift(&mut b, &sound, true);
        let after = shift(&mut b, &sound, false);
        let m = b.op("Mul", &[&space, &before], vec![]);
        let m = b.op("Mul", &[&m, &after], vec![]);
        let round = |b: &mut Nodes, x: &str| {
            let r = b.op("Div", &[x, "speed"], vec![]);
            b.op("Round", &[&r], vec![])
        };
        let top = round(&mut b, &p("limit"));
        let set = round(&mut b, &p("set"));
        let most = b.op("Max", &[&set, &p("2")], vec![]);
        // token i-1's frames (a), token i's (d); a gap is such a pair with a + d over the top
        let a = shift(&mut b, &raw, true);
        let pair = b.op("Add", &[&a, &raw], vec![]);
        let over = b.op("Greater", &[&pair, &top], vec![]);
        let over = f(&mut b, &over);
        let gap = b.op("Mul", &[&m, &over], vec![]);
        // set to most: a' = max(1, min(a, most - d)), d' = min(d, most - a')
        let room = b.op("Sub", &[&most, &raw], vec![]);
        let a2 = b.op("Min", &[&a, &room], vec![]);
        let a2 = b.op("Max", &[&a2, &p("1")], vec![]);
        let room = b.op("Sub", &[&most, &a2], vec![]);
        let d2 = b.op("Min", &[&raw, &room], vec![]);
        // each gap's new d at i, its new a moved back to i-1 (a gap's i-1 is never another gap's i)
        let gap_next = shift(&mut b, &gap, false);
        let a2_next = shift(&mut b, &a2, false);
        let dd = b.op("Sub", &[&d2, &raw], vec![]);
        let dd = b.op("Mul", &[&dd, &gap], vec![]);
        let da = b.op("Sub", &[&a2_next, &raw], vec![]);
        let da = b.op("Mul", &[&da, &gap_next], vec![]);
        let sum = b.op("Add", &[&raw, &dd], vec![]);
        b.new.push(node("Add", &[&sum, &da], &[&frames], vec![]));
        self.nodes.splice(at + 1..at + 1, b.new);
        Ok(())
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

/// The two speech models the patch takes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Model {
    Kokoro,
    Kerstin,
}

impl Model {
    /// The node that times each token, the limit on a word's last sound and the space after it, and the frames
    /// such a pair over it gets, both at speed 1 (plan, "Silence between words").
    fn timing(self) -> (&'static str, i64, i64) {
        match self {
            Model::Kokoro => ("/encoder/Clip", 8, 4),
            Model::Kerstin => ("/Clip", 10, 3),
        }
    }
}

/// Writes `src` patched to `dst` (through a .part file): Kokoro's STFT for the GPU, both models' limit on
/// the silence between words; `directml` adds the pads' change.
pub fn patch(src: &Path, dst: &Path, model: Model, directml: bool) -> Result<()> {
    rewrite(src, dst, |graph| {
        if model == Model::Kokoro {
            graph.gpufix()?;
        }
        let (clip, limit, set) = model.timing();
        graph.limit_word_gaps(clip, limit, set)?;
        if directml {
            let fixed = graph.padfix()?;
            if fixed == 0 || (model == Model::Kokoro && fixed != 3) {
                bail!("the model's upsampling is not the one expected");
            }
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

    /// Runs a graph made only of elementwise ops, Slice and Concat on one row (batch 1), as ONNX Runtime would.
    fn run(g: &Graph, feeds: &[(&str, Vec<f64>)]) -> std::collections::HashMap<String, Vec<f64>> {
        let mut t: std::collections::HashMap<String, Vec<f64>> =
            feeds.iter().map(|(k, v)| (k.to_string(), v.clone())).collect();
        for i in &g.inits {
            t.insert(str1(i, 8), values(i).unwrap());
        }
        let two = |a: &[f64], b: &[f64], f: fn(f64, f64) -> f64| -> Vec<f64> {
            let n = a.len().max(b.len());
            (0..n).map(|k| f(a[k % a.len()], b[k % b.len()])).collect()
        };
        for n in &g.nodes {
            let ins: Vec<Vec<f64>> = strs(n, 1).iter().filter(|x| !x.is_empty()).map(|x| t[x].clone()).collect();
            let out = match str1(n, 4).as_str() {
                "Cast" | "Identity" => ins[0].clone(),
                "Round" => ins[0].iter().map(|x| x.round_ties_even()).collect(),
                "Clip" => two(&ins[0], &ins[1], f64::max),
                "Equal" => two(&ins[0], &ins[1], |a, b| f64::from(a == b)),
                "GreaterOrEqual" => two(&ins[0], &ins[1], |a, b| f64::from(a >= b)),
                "Greater" => two(&ins[0], &ins[1], |a, b| f64::from(a > b)),
                "Add" => two(&ins[0], &ins[1], |a, b| a + b),
                "Sub" => two(&ins[0], &ins[1], |a, b| a - b),
                "Mul" => two(&ins[0], &ins[1], |a, b| a * b),
                "Div" => two(&ins[0], &ins[1], |a, b| a / b),
                "Min" => two(&ins[0], &ins[1], f64::min),
                "Max" => two(&ins[0], &ins[1], f64::max),
                "Concat" => ins.concat(),
                "Slice" => {
                    let len = ins[0].len() as f64;
                    let at = |x: f64| (if x < 0.0 { x + len } else { x }).clamp(0.0, len) as usize;
                    ins[0][at(ins[1][0])..at(ins[2][0])].to_vec()
                }
                op => panic!("{op} is not run here"),
            };
            t.insert(strs(n, 2)[0].clone(), out);
        }
        t
    }

    /// A duration graph as Kokoro's ends: Clip (min 1) and Cast, with input_ids and speed as inputs.
    fn durations(clip: &str) -> Graph {
        let info = |name: &str| V::Bytes(encode(&vec![(1, s(name))]));
        Graph {
            nodes: vec![
                [
                    vec![(3, s(clip))],
                    node("Clip", &["rounded", "one", ""], &["clipped"], vec![]),
                ]
                .concat(),
                node("Cast", &["clipped"], &["frames"], vec![int("to", INT64 as i64)]),
            ],
            inits: vec![tensor_f32("one", &[], &[1.0])],
            rest: vec![(11, info("input_ids")), (11, info("speed")), (12, info("frames"))],
        }
    }

    #[test]
    fn the_limit_sets_a_long_pair_between_words_and_leaves_the_rest() {
        let mut g = Graph::read(&durations("/encoder/Clip").write()).unwrap();
        g.limit_word_gaps("/encoder/Clip", 8, 4).unwrap();
        g.prune().unwrap();
        // pad, a word's end + space (15+5), a comma + space, a short pair (2+3), a pair the space
        // makes long (4+12), a double space, a pair at the limit (6+2), pad
        let ids = [0., 50., 16., 51., 3., 16., 52., 16., 53., 16., 54., 16., 16., 57., 16., 55., 0.];
        let raw = [1., 15., 5., 3., 9., 9., 2., 3., 4., 12., 6., 20., 20., 6., 2., 2., 1.];
        let frames = |speed: f64| {
            run(&g, &[("input_ids", ids.to_vec()), ("rounded", raw.to_vec()), ("speed", vec![speed])])["frames"].clone()
        };
        // a pair over 8 frames gets 4, its space keeping what the last sound cannot give; pairs at or under 8 stay
        assert_eq!(frames(1.0), [1., 1., 3., 3., 9., 9., 2., 3., 1., 3., 6., 20., 20., 6., 2., 2., 1.]);
        // both scale with speed: over Round(8 / 0.5) = 16 gets Round(4 / 0.5) = 8
        assert_eq!(frames(0.5), [1., 3., 5., 3., 9., 9., 2., 3., 4., 12., 6., 20., 20., 6., 2., 2., 1.]);
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
    fn kerstins_patch_limits_the_gaps_and_changes_the_pads_and_leaves_the_rest() {
        let dir = std::env::temp_dir().join(format!("ebook2m4b-pads-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (src, dst) = (dir.join("in.onnx"), dir.join("out.onnx"));
        let (mut g, d) = (tiny(), durations("/Clip"));
        g.nodes.extend(d.nodes);
        g.inits.extend(d.inits);
        g.rest.extend(d.rest);
        std::fs::write(&src, encode(&vec![(7, V::Bytes(g.write()))])).unwrap();
        patch(&src, &dst, Model::Kerstin, true).unwrap();
        let model = parse(&std::fs::read(&dst).unwrap()).unwrap();
        let V::Bytes(g) = &model[0].1 else { panic!() };
        let g = Graph::read(g).unwrap();
        let ops: Vec<String> = g.nodes.iter().map(|n| str1(n, 4)).collect();
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(ops.contains(&"STFT".into()) && !ops.contains(&"Conv".into()), "{ops:?}");
        assert!(g.nodes.iter().any(|n| strs(n, 2) == ["frames"] && str1(n, 4) == "Cast"));
        assert!(g.nodes.iter().any(|n| strs(n, 2) == ["clipped"] && str1(n, 4) == "Add")); // the limit's last
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
