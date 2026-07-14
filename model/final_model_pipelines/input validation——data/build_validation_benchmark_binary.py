"""Build a small binary benchmark for the input-validation layer.

Output labels are intentionally limited to:
  - success
  - fail

The script uses the local fake-job-posting CSV for genuine and fraudulent job
samples, then downloads public text/code corpora for non-job/invalid samples.

Fraudulent recruitment posts are included as valid inputs because this layer
only checks whether text can be sent to the fake-job classifier. It must not
reject job-related text just because it contains scam indicators.
"""

from __future__ import annotations

import csv
import gzip
import html
import io
import json
import random
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from xml.etree import ElementTree


BENCHMARK_DIR = Path(__file__).resolve().parent
PIPELINES_ROOT = BENCHMARK_DIR.parent
SOURCE_DATA_DIR = PIPELINES_ROOT / "data"
RAW_DIR = BENCHMARK_DIR / "validation_raw"
OUTPUT_PATH = BENCHMARK_DIR / "validation_benchmark_binary.csv"

SEED = 42
RANDOM = random.Random(SEED)

TARGET_COUNTS = {
    "valid_job": 500,
    "fake_recruitment_job": 100,
    "not_job_related_news": 300,
    "not_job_related_forum": 300,
    "invalid_chat": 200,
    "invalid_code": 200,
    "invalid_synthetic": 100,
}

URLS = {
    "ag_news": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "20news": "https://ndownloader.figshare.com/files/5975967",
    "nps_chat": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/nps_chat.zip",
}

CODESEARCHNET_CANDIDATES = (
    "https://huggingface.co/datasets/code_search_net/resolve/main/python/test-00000-of-00001.parquet",
    "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python/final/jsonl/test/python_test_0.jsonl.gz",
    "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript/final/jsonl/test/javascript_test_0.jsonl.gz",
    "https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java/final/jsonl/test/java_test_0.jsonl.gz",
)

WORD_RE = re.compile(r"[A-Za-z]+")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    value = html.unescape(value or "")
    value = NON_ASCII_RE.sub(" ", value)
    value = SPACE_RE.sub(" ", value).strip()
    return value


def is_english_text(text: str, min_words: int = 8) -> bool:
    words = WORD_RE.findall(text)
    if len(words) < min_words:
        return False
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return False
    letters = sum(ch.isalpha() for ch in compact)
    return letters / len(compact) >= 0.35


def download(url: str, filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=90) as response:
        path.write_bytes(response.read())
    return path


def take_unique(candidates: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    seen: set[str] = set()
    unique: list[dict[str, str]] = []
    RANDOM.shuffle(candidates)
    for row in candidates:
        key = row["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) == count:
            break
    if len(unique) < count:
        raise RuntimeError(f"Only collected {len(unique)} rows, need {count}.")
    return unique


def build_valid_jobs() -> list[dict[str, str]]:
    source_path = SOURCE_DATA_DIR / "cleaned_data.csv"
    rows: list[dict[str, str]] = []
    with source_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            if item.get("fraudulent") != "0":
                continue
            text = clean_text(item.get("combined_text", ""))
            if not text:
                text = clean_text(f"{item.get('description', '')} {item.get('requirements', '')}")
            if is_english_text(text, min_words=30):
                rows.append(
                    {
                        "source": "local_cleaned_data.csv",
                        "category": "valid_job",
                        "label": "success",
                        "text": text,
                    }
                )
    return take_unique(rows, TARGET_COUNTS["valid_job"])


def build_fake_recruitment_jobs() -> list[dict[str, str]]:
    source_path = SOURCE_DATA_DIR / "cleaned_data.csv"
    rows: list[dict[str, str]] = []
    with source_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for item in reader:
            if item.get("fraudulent") != "1":
                continue
            text = clean_text(item.get("combined_text", ""))
            if not text:
                text = clean_text(
                    " ".join(
                        item.get(field, "")
                        for field in (
                            "title",
                            "company_profile",
                            "description",
                            "requirements",
                            "benefits",
                        )
                    )
                )
            if is_english_text(text, min_words=30):
                rows.append(
                    {
                        "source": "local_cleaned_data.csv:fraudulent=1",
                        "category": "fake_recruitment_job",
                        "label": "success",
                        "text": text,
                    }
                )
    return take_unique(rows, TARGET_COUNTS["fake_recruitment_job"])


def build_ag_news() -> list[dict[str, str]]:
    path = download(URLS["ag_news"], "ag_news_train.csv")
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for item in reader:
            if len(item) < 3:
                continue
            text = clean_text(f"{item[1]} {item[2]}")
            if is_english_text(text, min_words=20):
                rows.append(
                    {
                        "source": "ag_news",
                        "category": "not_job_related_news",
                        "label": "fail",
                        "text": text,
                    }
                )
    return take_unique(rows, TARGET_COUNTS["not_job_related_news"])


def build_20_newsgroups() -> list[dict[str, str]]:
    path = download(URLS["20news"], "20news-bydate.tar.gz")
    rows: list[dict[str, str]] = []
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        RANDOM.shuffle(members)
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read().decode("latin-1", errors="ignore")
            body = raw.split("\n\n", 1)[-1]
            text = clean_text(body)
            if 80 <= len(text) <= 2500 and is_english_text(text, min_words=25):
                rows.append(
                    {
                        "source": "20_newsgroups",
                        "category": "not_job_related_forum",
                        "label": "fail",
                        "text": text,
                    }
                )
            if len(rows) >= TARGET_COUNTS["not_job_related_forum"] * 3:
                break
    return take_unique(rows, TARGET_COUNTS["not_job_related_forum"])


def build_nps_chat() -> list[dict[str, str]]:
    path = download(URLS["nps_chat"], "nps_chat.zip")
    rows: list[dict[str, str]] = []
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if not name.endswith(".xml"):
                continue
            root = ElementTree.fromstring(zf.read(name))
            for post in root.iter("Post"):
                text = clean_text("".join(post.itertext()))
                if 1 <= len(WORD_RE.findall(text)) <= 12 and text:
                    rows.append(
                        {
                            "source": "nps_chat",
                            "category": "invalid_chat",
                            "label": "fail",
                            "text": text,
                        }
                    )
    return take_unique(rows, TARGET_COUNTS["invalid_chat"])


def try_load_codesearchnet(url: str) -> list[dict[str, str]]:
    filename = url.rsplit("/", 1)[-1]
    path = download(url, filename)
    rows: list[dict[str, str]] = []
    if filename.endswith(".parquet"):
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError("Reading CodeSearchNet parquet requires polars.") from exc
        frame = pl.read_parquet(path, columns=["func_code_string"])
        for item in frame["func_code_string"].to_list():
            code = clean_text(str(item))
            if "def " in code or "function " in code or "class " in code or "import " in code:
                if len(code) >= 40:
                    rows.append(
                        {
                            "source": "codesearchnet",
                            "category": "invalid_code",
                            "label": "fail",
                            "text": code[:3000],
                        }
                    )
            if len(rows) >= TARGET_COUNTS["invalid_code"] * 2:
                break
        return rows

    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            item = json.loads(line)
            code = clean_text(item.get("code", ""))
            if "def " in code or "function " in code or "class " in code or "import " in code:
                if len(code) >= 40:
                    rows.append(
                        {
                            "source": "codesearchnet",
                            "category": "invalid_code",
                            "label": "fail",
                            "text": code[:3000],
                        }
                    )
            if len(rows) >= TARGET_COUNTS["invalid_code"] * 2:
                break
    return rows


def build_codesearchnet() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for url in CODESEARCHNET_CANDIDATES:
        try:
            rows.extend(try_load_codesearchnet(url))
            if len(rows) >= TARGET_COUNTS["invalid_code"]:
                break
        except (urllib.error.URLError, OSError, EOFError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")
    if len(rows) < TARGET_COUNTS["invalid_code"]:
        raise RuntimeError("Could not collect enough CodeSearchNet rows:\n" + "\n".join(errors))
    return take_unique(rows, TARGET_COUNTS["invalid_code"])


def build_synthetic() -> list[dict[str, str]]:
    samples = [
        "",
        " ",
        "https://example.com",
        "www.example.com/job",
        "https://a.com https://b.org",
        "1234567890",
        "!!! ??? ### $$$",
        "asdf qwer zxcv",
        "xqz brr tsk nth",
        "lol",
        "hi",
        "hello",
        "ok",
        "thank you",
        "test test",
        "qwerty",
        "9 8 7 6 5 4 3 2 1",
        "@@@@ #### !!!!",
        "http://localhost:3000",
        "foo_bar_baz_12345",
    ]
    while len(samples) < TARGET_COUNTS["invalid_synthetic"]:
        token = "".join(RANDOM.choice("bcdfghjklmnpqrstvwxyz0123456789!@#$%") for _ in range(RANDOM.randint(8, 28)))
        samples.append(token)
    return [
        {
            "source": "synthetic",
            "category": "invalid_synthetic",
            "label": "fail",
            "text": text,
        }
        for text in samples[: TARGET_COUNTS["invalid_synthetic"]]
    ]


def main() -> None:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        build_valid_jobs,
        build_fake_recruitment_jobs,
        build_ag_news,
        build_20_newsgroups,
        build_nps_chat,
        build_codesearchnet,
        build_synthetic,
    ]
    rows: list[dict[str, str]] = []
    for build in builders:
        part = build()
        print(f"{part[0]['category']}: {len(part)}")
        rows.extend(part)

    for idx, row in enumerate(rows, start=1):
        row["id"] = str(idx)

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "source", "label", "text"])
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["category"], row["label"])
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    for (category, label), count in sorted(counts.items()):
        print(f"{category},{label},{count}")


if __name__ == "__main__":
    main()
