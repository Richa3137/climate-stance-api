"""
API Client: Input CSV -> Render NLP API -> Output CSV
=====================================================
This is the "API import" part of the professor's workflow:
  1. Read the Input CSV (id, text)
  2. Send the texts to the NLP API hosted on Render (POST /predict)
  3. Receive the predicted labels
  4. Save the Output CSV (id, label)

Usage (from a terminal):
  python api_client.py --api-url https://YOUR-SERVICE.onrender.com --input sample_input.csv --output output_predictions.csv

Usage (from Python):
  from api_client import predict_csv_via_render_api
  predict_csv_via_render_api("sample_input.csv", api_url="https://YOUR-SERVICE.onrender.com")
"""
import argparse
import io
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

DEFAULT_API_URL = os.environ.get("RENDER_API_URL", "http://127.0.0.1:8000")
ID_NAMES = ["ID", "INDEX", "ROW", "ROW_ID", "S.NO", "SNO", "SR.NO"]
TEXT_NAMES = ["text", "clean_comment", "hindi_comment", "raw_comment", "comment", "tweet", "sentence", "content", "post"]


def wait_until_ready(api_url: str, max_wait_seconds: int = 420) -> dict:
    """
    Render's free plan puts the server to sleep when it is idle. The first request wakes it up,
    and it then downloads the model from Hugging Face. This waits until /health says 'ready'.
    """
    api_url = api_url.rstrip("/")
    start = time.time()
    last_msg = ""
    while time.time() - start < max_wait_seconds:
        try:
            r = requests.get(f"{api_url}/health", timeout=30)
            if r.status_code == 200:
                info = r.json()
                status = info.get("status")
                if status == "ready":
                    return info
                if status == "error":
                    raise RuntimeError(f"The API could not load the model: {info.get('error')}")
                msg = "API is up, model is still loading from Hugging Face..."
            else:
                msg = f"API answered with HTTP {r.status_code}, waiting..."
        except requests.RequestException:
            msg = "Waking up the Render server (this can take a minute on the free plan)..."
        if msg != last_msg:
            print(f"  {msg}")
            last_msg = msg
        time.sleep(5)
    raise TimeoutError(f"API at {api_url} was not ready after {max_wait_seconds} seconds.")


def predict_batch_via_api(
    texts: List[str],
    api_url: Optional[str] = None,
    batch_size: int = 32,
    timeout: float = 300.0,
) -> Tuple[List[str], List[float], np.ndarray]:
    """Sends texts to POST /predict. Returns (labels, confidences, probabilities[AGAINST, NONE, FAVOR])."""
    base = (api_url or DEFAULT_API_URL).rstrip("/")
    labels: List[str] = []
    confs: List[float] = []
    probs: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        chunk = [str(t) for t in texts[start:start + batch_size]]
        payload = {"data": [{"id": start + i + 1, "text": t} for i, t in enumerate(chunk)]}
        r = requests.post(f"{base}/predict", json=payload, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"API request failed (HTTP {r.status_code}): {r.text[:300]}")
        for item in r.json()["predictions"]:
            labels.append(item["label"])
            confs.append(float(item["confidence"]))
            p = item["probabilities"]
            probs.append([float(p["AGAINST"]), float(p["NONE"]), float(p["FAVOR"])])
        print(f"  Predicted {min(start + batch_size, len(texts))}/{len(texts)} rows")
    return labels, confs, np.array(probs, dtype=np.float32)


def load_input_file(path: str) -> Tuple[pd.DataFrame, str, str]:
    """Reads a CSV / TSV / TXT / XLSX file and finds the id column and the text column."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    low = path.lower()
    if low.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        raw = open(path, "rb").read()
        content = None
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                content = raw.decode(enc)
                break
            except Exception:
                continue
        if content is None:
            content = raw.decode("utf-8", errors="replace")
        first_line = content.splitlines()[0] if content.splitlines() else ""
        if low.endswith(".txt") and "\t" not in first_line:
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            return pd.DataFrame({"id": range(1, len(lines) + 1), "text": lines}), "id", "text"
        sep = "\t" if (low.endswith((".tsv", ".txt")) or ("\t" in first_line and "," not in first_line)) else ","
        df = pd.read_csv(io.StringIO(content), sep=sep)

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    id_col = next((c for c in df.columns if c.strip().upper() in ID_NAMES), None)
    if id_col is None:
        df.insert(0, "id", range(1, len(df) + 1))
        id_col = "id"
    text_col = next((c for c in df.columns if c.strip().lower() in TEXT_NAMES), None)
    if text_col is None:
        candidates = [c for c in df.columns if c != id_col and not pd.api.types.is_numeric_dtype(df[c])]
        candidates = candidates or [c for c in df.columns if c != id_col]
        text_col = max(candidates, key=lambda c: df[c].astype(str).str.len().mean())
    return df, id_col, text_col


def predict_csv_via_render_api(
    input_csv_path: str,
    output_csv_path: str = "output_predictions.csv",
    api_url: Optional[str] = None,
) -> pd.DataFrame:
    base = (api_url or DEFAULT_API_URL).rstrip("/")
    df, id_col, text_col = load_input_file(input_csv_path)

    print("=" * 70)
    print("INPUT CSV  ->  NLP API ON RENDER  ->  OUTPUT CSV")
    print("=" * 70)
    print(f"Input file : {os.path.abspath(input_csv_path)} ({len(df)} rows)")
    print(f"Columns    : id = '{id_col}', text = '{text_col}'")
    print(f"API        : {base}")

    info = wait_until_ready(base)
    print(f"Model      : {info.get('model_id')} (status: {info.get('status')})")

    labels, _, _ = predict_batch_via_api(df[text_col].fillna("").astype(str).tolist(), api_url=base)

    out = pd.DataFrame({"id": df[id_col].tolist(), "label": labels})
    out.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print("-" * 70)
    print(f"Saved Output CSV (id, label): {os.path.abspath(output_csv_path)}")
    print(out.head(15).to_string(index=False))
    print("=" * 70)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send an Input CSV (id, text) to the Render NLP API and save an Output CSV (id, label).")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Your Render URL, e.g. https://climastance-nlp-api.onrender.com")
    parser.add_argument("--input", default="sample_input.csv", help="Input CSV with id and text columns")
    parser.add_argument("--output", default="output_predictions.csv", help="Where to save the Output CSV (id, label)")
    args = parser.parse_args()
    predict_csv_via_render_api(args.input, args.output, args.api_url)
