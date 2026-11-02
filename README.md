# ebook2m4b

Turns English and German ebooks into audiobooks on your graphics card.

1. Reads mobi, azw3, epub, fb2, docx, txt
and html
2. Cleans the text
3. Speaks it with the Kokoro voice model (German books: the voice Kerstin, picked up automatically)
4. Checks the result with Whisper
5. Writes one M4B file per book, with chapters

## Requirements

- About 10 GB of free disk for the tools and models
- Windows 10 or 11, 64-bit, with an NVIDIA card (CUDA) or another card (DirectML and Vulkan).
- Linux, x86-64, with an NVIDIA card and its driver (other cards are not supported on Linux); the
  download needs glibc 2.39 or later (Ubuntu 24.04, Debian 13 or newer).


## Quickstart

Download the [latest build](https://github.com/elana-voss/ebook2m4b/releases/latest), unpack it and run:

```
./ebook2m4b samples/alice.epub
```

The audiobook is written to the `out` folder as alice.m4b. Work in progress is kept in the `work` folder, an interrupted run goes on where it stopped.

The first run downloads the tools and models it needs with an NVIDIA card:
* About 8.4 GB on Windows (9.7 GB unpacked) into  `%LOCALAPPDATA%\ebook2m4b`.
* About 8.1 GB on Linux (9.4 GB unpacked) into `~/.local/share/ebook2m4b`. 

The language is told from the book's text. The first German book downloads the German voice, Kerstin, about 338 MB more. A book in neither English nor German is reported as not done and the run goes on.


## Examples

A short demo, about a minute each from the start, middle and end of the book, written to the out folder
as alice.demo.m4a:

```
./ebook2m4b samples/alice.epub --demo
```

Another voice (Kokoro's English voices start with `af_`, `am_`, `bf_` or `bm_`; the default is
`af_bella`; a German book is always read by Kerstin):

```
./ebook2m4b samples/alice.epub --voice bm_george
```

A little slower:

```
./ebook2m4b samples/alice.epub --speed 0.9
```

Every book in a folder, into another output folder:

```
./ebook2m4b ebooks --out audiobooks
```

A finished book is skipped; to make it again:

```
./ebook2m4b samples/alice.epub --overwrite
```

To list the options:

```
./ebook2m4b --help
```

## Building from source

You need Rust, the Visual Studio C++ build tools, CMake, Git, Strawberry Perl, LLVM's libclang and the
Vulkan SDK. Then, from this folder:

```
set LIBCLANG_PATH=C:\path\to\libclang
set VULKAN_SDK=C:\VulkanSDK\<version>
cargo build --release
```

Put Strawberry Perl's `perl\bin` first on PATH, but not its `c\bin`. The program is
`target\release\ebook2m4b.exe`.

On Linux you need gcc and g++, CMake, clang (libclang), perl, make, curl, pkg-config and OpenSSL's
headers (Debian's libssl-dev); `tools/linux_build_env.sh` fetches Rust without root. Then:

```
bash tools/linux_build_env.sh
source ~/ebook2m4b-build/env.sh
cargo build --release
```

The program is `target/release/ebook2m4b`.

## Sample book

`samples/alice.epub` is *Alice's Adventures in Wonderland* by Lewis Carroll, Project Gutenberg eBook
#11 (https://www.gutenberg.org/ebooks/11), in the public domain in the USA. It is included unchanged,
with the Project Gutenberg License it carries (https://www.gutenberg.org/policy/license.html).

## License

GPL-3.0. See `LICENSE`.
