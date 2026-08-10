#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_FILE="$DATA_DIR/raw/emscad_v1.csv"
EXPECTED_SHA256="788305f8bab0325191014887fba99e56515825f757d300529059ccf3bd0ba033"
DATASET_URL="${EMSCAD_DATASET_URL:-${1:-}}"

if [[ -z "$DATASET_URL" ]]; then
  echo "Usage: EMSCAD_DATASET_URL=https://.../emscad_v1.csv $0" >&2
  echo "       $0 https://.../emscad_v1.csv" >&2
  exit 2
fi

mkdir -p "$DATA_DIR/raw"
TEMP_FILE="$(mktemp "$DATA_DIR/raw/emscad_v1.csv.XXXXXX")"
trap 'rm -f "$TEMP_FILE"' EXIT

curl --fail --location --retry 3 --output "$TEMP_FILE" "$DATASET_URL"
ACTUAL_SHA256="$(shasum -a 256 "$TEMP_FILE" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "Checksum mismatch for downloaded EMSCAD dataset." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual:   $ACTUAL_SHA256" >&2
  exit 1
fi

mv "$TEMP_FILE" "$RAW_FILE"
trap - EXIT

python3 "$DATA_DIR/raw/split_paper_aligned_data.py" \
  --raw "$RAW_FILE" \
  --out "$DATA_DIR/splits" \
  --also_save_processed

echo "Dataset verified and generated under $DATA_DIR"
