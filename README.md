# ebook2m4b

Turns English ebooks into audiobooks on your own computer. It reads mobi, azw3, epub, fb2, docx, txt
and html, cleans the text, speaks it with the Kokoro voice model, checks the result with Whisper and
writes one M4B file per book, with chapters.

## Requirements

- Windows 10 or 11, 64-bit.
- A graphics card: NVIDIA (CUDA), or another card through DirectML and Vulkan.
- About 10 GB of free disk for the tools and models (see Quickstart).

## Build

You need Rust, the Visual Studio C++ build tools, CMake, Git, Strawberry Perl, LLVM's libclang and the
Vulkan SDK. Then, from this folder:

```
set LIBCLANG_PATH=C:\path\to\libclang
set VULKAN_SDK=C:\VulkanSDK\<version>
cargo build --release
```

Put Strawberry Perl's `perl\bin` first on PATH, but not its `c\bin`. The program is
`target\release\ebook2m4b.exe`.

## Quickstart

```
ebook2m4b samples\alice.epub
```

The first run downloads the tools and models it needs, about 8.4 GB with an NVIDIA card (9.7 GB
once unpacked), into `%LOCALAPPDATA%\ebook2m4b`. It shows each download with its size and its
progress in percent. Later runs find everything there and start at once.

The audiobook is written to `out\alice.m4b`; work in progress is kept in `work\`, so an interrupted
run goes on where it stopped.

## Examples

A short demo, about a minute each from the start, middle and end of the book, written to
`out\alice.demo.m4a`:

```
ebook2m4b samples\alice.epub --demo
```

Another voice (Kokoro's English voices start with `af_`, `am_`, `bf_` or `bm_`; the default is
`af_bella`):

```
ebook2m4b samples\alice.epub --voice bm_george
```

A little slower:

```
ebook2m4b samples\alice.epub --speed 0.9
```

Every book in a folder, into another output folder:

```
ebook2m4b D:\ebooks --out D:\audiobooks
```

A finished book is skipped; to make it again:

```
ebook2m4b samples\alice.epub --overwrite
```

`ebook2m4b --help` lists the options.

## Sample book

`samples\alice.epub` is *Alice's Adventures in Wonderland* by Lewis Carroll, Project Gutenberg eBook
#11 (https://www.gutenberg.org/ebooks/11), in the public domain in the USA. It is included unchanged,
with the Project Gutenberg License it carries (https://www.gutenberg.org/policy/license.html).

## License

GPL-3.0. See `LICENSE`.
