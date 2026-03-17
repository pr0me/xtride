# DIRT Dataset Extractor

Rust utility for extracting selected files from the DIRT dataset archives into a flattened output layout.

## What it does

The extractor scans a directory for `*.tar.xz` archives, reads matching `bins/` and `types/` entries, and writes them into an output directory without the per-bucket subdirectories.

It supports:

- extracting all matching archive contents
- restricting extraction to hashes listed in a file
- adding a random sample of extra hashes
- writing the extracted hash list to `hashes_out.txt`
- parallel archive processing with progress reporting

## Build

```bash
cargo build --release
```

## Usage

```bash
cargo run --release -- --directory /path/to/dataset
```

Useful options:

- `--output <dir>`: choose the output directory, default `extracted`
- `--threads <n>`: override the default thread count
- `--hash-file <path>`: extract only hashes listed in a file
- `--add <n>`: add `n` randomly sampled hashes in addition to `--hash-file`
- `--skip-extraction`: skip archive extraction
- `--verbose`: print additional runtime information

## Input layout

The tool expects archives containing paths like:

```text
bins/<bucket>/<hash>.jsonl.gz
types/<bucket>/<hash>.json.gz
```

## Output layout

The output directory contains:

- `bins/<hash>.jsonl.gz`
- `types/<hash>.json.gz`
- `hashes_out.txt`