# Dataset handling
Shared EMSCAD data for all sprint3 models.
## Folders
- **`raw/`**: Holds the original EMSCAD file (`emscad_v1.csv`) and the paper-aligned split script.
- **`splits/`**: Stores the fixed train / validation / test files used by every sprint3 model.
- **`processed/`**: Stores the cleaned Condition-A combined-text table generated when `--also_save_processed` is used.
- **`samples/`**: Holds a tiny EMSCAD excerpt for quick local checks.
## Code
- **`raw/split_paper_aligned_data.py`**: Cleans EMSCAD text, builds `combined_text` / `model_text`, and writes the stratified 80/20 + 10%-of-train validation splits.
Download helper:
```bash
EMSCAD_DATASET_URL="https://approved-source/emscad_v1.csv" ./download_data.sh
```