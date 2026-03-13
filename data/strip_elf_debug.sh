#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <input-dir>" >&2
  exit 1
fi

input_dir="$1"

if [[ ! -d "$input_dir" ]]; then
  echo "error: not a directory: $input_dir" >&2
  exit 1
fi

strip_bin="/opt/homebrew/opt/llvm/bin/llvm-strip"
if [[ ! -x "$strip_bin" ]]; then
  echo "error: llvm-strip not found or not executable at $strip_bin" >&2
  exit 1
fi

# iterate regular files in the directory (non-recursive)
for file_path in "$input_dir"/*; do
  [[ -e "$file_path" ]] || continue
  [[ -f "$file_path" ]] || continue

  # skip files already suffixed with .strip
  if [[ "$file_path" == *.strip ]]; then
    continue
  fi

  # detect ELF via `file`
  file_type="$(file -b "$file_path" || true)"
  if [[ "$file_type" == ELF* ]]; then
    dest="${file_path}.strip"

    if ! cp -f -p "$file_path" "$dest"; then
      echo "error: failed to copy $file_path to $dest" >&2
      continue
    fi

    if ! "$strip_bin" --strip-unneeded "$dest"; then
      echo "error: strip failed for $dest" >&2
      continue
    fi

    echo "stripped debug: $dest"
  fi
done


