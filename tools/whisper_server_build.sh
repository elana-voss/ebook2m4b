#!/usr/bin/env bash
# whisper.cpp's server for Linux with CUDA, the program setup fetches there (src/downloads.rs): upstream
# publishes a CUDA build for Windows only. Usage: whisper_server_build.sh <whisper.cpp's tag> [work folder].
# Run by .github/workflows/whisper-server.yml. Without root it fetches NVIDIA's CUDA compiler (nvcc, cudart,
# culibos, crt, nvvm, cccl) and cuBLAS (the same versions as setup's libraries) and whisper.cpp at the tag;
# needs cmake, gcc/g++, make, curl, xz, git, python3. All of whisper.cpp goes into the one program; of CUDA it
# needs libcudart, libcublas and libcublasLt, from setup. Not built for this machine's processor (GGML_NATIVE),
# no OpenMP (libgomp is not on every system). Leaves in the work folder <name>.tar.gz and <name>.tar.gz.sha256.
set -euo pipefail
TAG=$1
B=${2:-$HOME/whisper-server-build}
# Another tag's checkout left here would be built and packed under this tag's name.
if [ -d "$B/src" ] && [ "$(git -C "$B/src" describe --tags --exact-match 2>/dev/null)" != "$TAG" ]; then
    echo "$B/src is another checkout than $TAG: give another work folder" >&2
    exit 1
fi
R=https://developer.download.nvidia.com/compute/cuda/redist
V=13.4.92
mkdir -p "$B/dl" "$B/cuda"
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
(cd "$B/cuda/lib" && for l in libcublas libcublasLt; do ln -sf "$l.so.13" "$l.so"; done)
[ -e "$B/cuda/lib64" ] || ln -s lib "$B/cuda/lib64"
export PATH="$B/cuda/bin:$PATH" CUDA_PATH="$B/cuda" CUDAToolkit_ROOT="$B/cuda" CUDACXX="$B/cuda/bin/nvcc"
export LIBRARY_PATH="$B/cuda/lib:$B/cuda/lib/stubs:${LIBRARY_PATH:-}"
[ -d "$B/src" ] || git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/whisper.cpp "$B/src"
cmake -S "$B/src" -B "$B/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_SERVER=ON -DWHISPER_BUILD_TESTS=OFF
cmake --build "$B/build" --target whisper-server -j 4
name=whisper-server-$TAG-linux-x64
rm -rf "${B:?}/$name"
mkdir "$B/$name"
cp "$B/build/bin/whisper-server" "$B/src/LICENSE" "$B/$name"
strip "$B/$name/whisper-server"
tar -C "$B" -czf "$B/$name.tar.gz" "$name"
(cd "$B" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
