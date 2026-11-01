#!/usr/bin/env bash
# The Linux build's tools, without root: Rust (rustup, minimal) and ~/ebook2m4b-build/env.sh, which puts it on
# PATH and names libclang. Needs cmake, gcc/g++, clang (libclang), perl, make, curl, pkg-config, OpenSSL's headers
# (libssl-dev; ort's build needs them). Then: `source ~/ebook2m4b-build/env.sh; cargo build --release -j 4`.
set -euo pipefail
B=~/ebook2m4b-build
mkdir -p "$B"
command -v cargo >/dev/null || [ -x ~/.cargo/bin/cargo ] ||
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
cat > "$B/env.sh" <<EOT
export PATH="$HOME/.cargo/bin:\$PATH"
export LIBCLANG_PATH=\$(dirname \$(ls /usr/lib/llvm-*/lib/libclang*.so.* | head -1))
EOT
echo "ready: source $B/env.sh"
