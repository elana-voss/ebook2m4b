#!/usr/bin/env bash
# The Linux build's tools, without root, into ~/ebook2m4b-build: Rust (rustup, minimal) and the CUDA
# compiler from NVIDIA's redistributable archives (nvcc, cudart, culibos, crt, nvvm, cccl) plus cuBLAS's headers and
# library from NVIDIA's wheel, for whisper-rs's "cuda" feature (Cargo.toml, Linux). Needs cmake, gcc/g++,
# clang (libclang), perl, make, curl, xz, pkg-config, OpenSSL's headers (libssl-dev; ort's build needs them). Then: `source ~/ebook2m4b-build/env.sh; cargo build --release -j 4`.
set -euo pipefail
B=~/ebook2m4b-build
R=https://developer.download.nvidia.com/compute/cuda/redist
V=13.4.92
mkdir -p "$B/dl" "$B/cuda"
command -v cargo >/dev/null || [ -x ~/.cargo/bin/cargo ] ||
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
for p in cuda_nvcc cuda_cudart cuda_culibos cuda_crt libnvvm cccl; do
    ver=$V; [ "$p" = cccl ] && ver=$(curl -s "$R/redistrib_13.4.3.json" | python3 -c "import json,sys;print(json.load(sys.stdin)['cccl']['version'])")
    f="$p-linux-x86_64-$ver-archive.tar.xz"
    [ -f "$B/dl/$f" ] || curl -sSfL -o "$B/dl/$f" "$R/$p/linux-x86_64/$f"
    tar -xJf "$B/dl/$f" -C "$B/cuda" --strip-components=1
done
w=nvidia_cublas-13.4.1.3-py3-none-manylinux_2_27_x86_64.whl
[ -f "$B/dl/$w" ] || curl -sSfL -o "$B/dl/$w" "https://files.pythonhosted.org/packages/e1/97/d483a41da34df5c88727c57319576967f6d7befba5028d70928d2999635e/$w"
python3 - "$B/dl/$w" "$B/cuda" <<'EOF'
import os, sys, zipfile
z, dest = zipfile.ZipFile(sys.argv[1]), sys.argv[2]
for n in z.namelist():
    for part in ("/include/", "/lib/"):
        if part in n and not n.endswith("/"):
            out = os.path.join(dest, part.strip("/"), n.split(part, 1)[1])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(z.read(n))
EOF
cd "$B/cuda/lib"
for l in libcublas libcublasLt; do ln -sf "$l.so.13" "$l.so"; done
cd "$B/cuda"; [ -e lib64 ] || ln -s lib lib64
cat > "$B/env.sh" <<EOF
export PATH="$HOME/.cargo/bin:$B/cuda/bin:\$PATH"
export CUDA_PATH="$B/cuda" CUDAToolkit_ROOT="$B/cuda" CUDACXX="$B/cuda/bin/nvcc"
export LIBRARY_PATH="$B/cuda/lib:$B/cuda/lib/stubs:\${LIBRARY_PATH:-}"
export LIBCLANG_PATH=\$(dirname \$(ls /usr/lib/llvm-*/lib/libclang*.so.* | head -1))
EOF
echo "ready: source $B/env.sh"
