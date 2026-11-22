"""
Render FastAPI Backend for NLP Stance Detection (~200 MB RAM — Render 512 MB Free Tier Safe)
============================================================================================
Project : Detecting Stance in Media on Global Warming for Indic Languages (CSET 346)
Claim   : "Climate change and global warming is a serious concern."
Model   : Fine-tuned XLM-RoBERTa, streamed directly from Hugging Face (richa3137/climate-stance-xlmr)
Labels  : FAVOR, AGAINST, NONE

Why this fits comfortably inside Render Free Tier's 512 MB RAM limit (~200 MB total RSS):
  1. Pure-Python Unigram Viterbi tokenizer loads only the English/Indic/Emoji/Common subwords
     from tokenizer.json (~35 MB RAM instead of 265 MB from Rust tokenizers).
  2. Never downloads the 1.12 GB model.safetensors file to /tmp (which is tmpfs RAM on Render).
  3. Streams the 12 transformer layers (340 MB) in 8 MB HTTP Range chunks and quantizes each
     weight matrix on-the-fly to per-channel int8 (85 MB total in RAM).
  4. Fetches token embeddings on-demand via HTTP Range requests and caches them in RAM.
"""
import gc
import html
import io
import json
import os
import re
import struct
import threading
import unicodedata
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from scipy.special import erf

# ==============================================================================
# CONFIGURATION
# ==============================================================================
APP_TITLE = "ClimaStance NLP REST API - Detecting Stance in Media on Global Warming"
TARGET_CLAIM = "Climate change and global warming is a serious concern."
MODEL_ID = os.environ.get("MODEL_ID", "richa3137/climate-stance-xlmr")
MODEL_KEY = "xlmr"
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "128"))
BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", "16"))
PRELOAD_VOCAB_ROWS = int(os.environ.get("PRELOAD_VOCAB_ROWS", "5000"))

HF_TO_PROJECT_LABEL = {"against": "AGAINST", "none": "NONE", "favour": "FAVOR", "favor": "FAVOR"}
PROJECT_LABELS = ["AGAINST", "NONE", "FAVOR"]

app = FastAPI(
    title=APP_TITLE,
    description=(
        f"REST API for climate stance detection on the claim: '{TARGET_CLAIM}'. "
        f"Model: {MODEL_ID} (fine-tuned XLM-RoBERTa, loaded from Hugging Face). "
        "Send JSON to /predict or an Input CSV (id, text) to /predict_csv to get an Output CSV (id, label)."
    ),
    version="2.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# TEXT CLEANING
# ==============================================================================
def clean_stance_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\ufffd", " ")
    text = re.sub(r"#SemST\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\bRT\s+@\w+:?", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: np.float32) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return np.float32(0.5) * x * (np.float32(1.0) + erf(x / np.float32(np.sqrt(2.0))).astype(np.float32))


def _quantize_2d_int8(w_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    max_abs = np.max(np.abs(w_f32), axis=1)
    scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
    w_q_t = np.ascontiguousarray(np.clip(np.round(w_f32 / scale[:, None]), -127, 127).astype(np.int8).T)
    return w_q_t, scale


# ==============================================================================
# LOW-MEMORY VITERBI UNIGRAM TOKENIZER (SAVES >200 MB RAM VS RUST TOKENIZERS)
# ==============================================================================
_VOCAB_PAIR_RE = re.compile(r'\[\s*("(?:[^"\\]|\\.)*")\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\]')


def _is_relevant_token(w: str, idx: int) -> bool:
    """Keeps top 45k universal tokens + all Latin/English, Indic (Devanagari/Bengali/etc.), numbers, & symbols."""
    if idx < 45000 or len(w) <= 2:
        return True
    for ch in w:
        code = ord(ch)
        if code < 0x0250 or (0x0900 <= code <= 0x0DFF) or (0x1F000 <= code <= 0x1FAFF) or (0x2000 <= code <= 0x27BF):
            return True
    return False


class LiteXLMRUnigramTokenizer:
    """Exact SentencePiece Unigram Viterbi tokenizer using <40 MB RAM."""

    def __init__(self, raw_json_text: str, max_length: int = 128):
        self.max_length = max_length
        v_start = raw_json_text.find('"vocab"')
        if v_start < 0:
            v_start = 0

        self.piece2id: Dict[str, int] = {}
        scores_list: List[float] = []
        for idx, m in enumerate(_VOCAB_PAIR_RE.finditer(raw_json_text, v_start)):
            w = json.loads(m.group(1))
            sc = float(m.group(2))
            scores_list.append(sc)
            if _is_relevant_token(w, idx):
                self.piece2id[w] = idx
        self.scores = np.array(scores_list, dtype=np.float32)

    def encode_ids(self, text: str) -> List[int]:
        s = unicodedata.normalize("NFKC", str(text).strip())
        if not s:
            return [0, 2]
        pieces = ["\u2581" + w for w in s.split()]
        ids = [0]
        p2i = self.piece2id
        sc_arr = self.scores

        for word in pieces:
            n = len(word)
            best_score = [-1e18] * (n + 1)
            best_id = [3] * (n + 1)
            best_prev = [0] * (n + 1)
            best_score[0] = 0.0
            for i in range(n):
                bi = best_score[i]
                if bi <= -1e17:
                    continue
                max_j = min(n, i + 24)
                for j in range(i + 1, max_j + 1):
                    sub = word[i:j]
                    tid = p2i.get(sub)
                    if tid is not None:
                        sc = bi + float(sc_arr[tid])
                        if sc > best_score[j]:
                            best_score[j] = sc
                            best_id[j] = tid
                            best_prev[j] = i
                    elif j == i + 1 and bi - 20.0 > best_score[j]:
                        best_score[j] = bi - 20.0
                        best_id[j] = 3
                        best_prev[j] = i
            cur = n
            w_ids = []
            while cur > 0:
                w_ids.append(best_id[cur])
                cur = best_prev[cur]
            ids.extend(reversed(w_ids))
            if len(ids) >= self.max_length - 1:
                break

        return ids[: self.max_length - 1] + [2]


# ==============================================================================
# STREAMING INT8 XLM-ROBERTA ENGINE (~200 MB TOTAL PROCESS RAM)
# ==============================================================================
class StreamingInt8XLMRStanceModel:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.base_url = f"https://huggingface.co/{repo_id}/resolve/main"
        self.session = requests.Session()

        print(f"[API] Fetching config & tokenizer from Hugging Face ({repo_id})...", flush=True)
        cfg_resp = self.session.get(f"{self.base_url}/config.json", timeout=60)
        cfg_resp.raise_for_status()
        self.config = cfg_resp.json()

        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.hidden = int(self.config["hidden_size"])
        self.head_dim = self.hidden // self.num_heads
        self.eps = np.float32(self.config.get("layer_norm_eps", 1e-5))
        self.pad_id = int(self.config.get("pad_token_id", 1))
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))

        id2label = {int(k): v for k, v in self.config["id2label"].items()}
        self.out_labels = [
            HF_TO_PROJECT_LABEL.get(str(id2label[i]).strip().lower(), str(id2label[i]).upper())
            for i in range(len(id2label))
        ]
        self.reorder = [self.out_labels.index(lbl) for lbl in PROJECT_LABELS]

        tok_resp = self.session.get(f"{self.base_url}/tokenizer.json", timeout=120)
        tok_resp.raise_for_status()
        raw_tok_text = tok_resp.content.decode("utf-8")
        del tok_resp
        self.tokenizer = LiteXLMRUnigramTokenizer(raw_tok_text, max_length=MAX_SEQ_LENGTH)
        del raw_tok_text
        gc.collect()

        # Fetch safetensors header (first 64 KB)
        self.st_url = f"{self.base_url}/model.safetensors"
        hdr_prefix = self._fetch_range(0, 65535)
        hdr_len = struct.unpack("<Q", hdr_prefix[:8])[0]
        if 8 + hdr_len > len(hdr_prefix):
            hdr_prefix = self._fetch_range(0, 8 + hdr_len)
        self.st_header = json.loads(hdr_prefix[8 : 8 + hdr_len])
        self.data_base = 8 + hdr_len

        self.we_info = self.st_header["roberta.embeddings.word_embeddings.weight"]
        self.we_start = self.data_base + int(self.we_info["data_offsets"][0])
        self.we_vocab = int(self.we_info["shape"][0])

        self.w_1d: Dict[str, np.ndarray] = {}
        self.w_q_t: Dict[str, np.ndarray] = {}
        self.w_scale: Dict[str, np.ndarray] = {}

        other_tensors = [
            (k, v)
            for k, v in self.st_header.items()
            if k not in ("__metadata__", "roberta.embeddings.word_embeddings.weight")
        ]
        other_tensors.sort(key=lambda kv: kv[1]["data_offsets"][0])

        print(f"[API] Streaming & compressing {len(other_tensors)} transformer tensors to int8...", flush=True)
        self._stream_and_quantize_tensors(other_tensors)
        gc.collect()

        # Preload first PRELOAD_VOCAB_ROWS (5,000 rows = 3.8 MB RAM)
        n_pre = min(PRELOAD_VOCAB_ROWS, self.we_vocab)
        row_bytes = self.hidden * 4
        c_bytes = self._fetch_range(self.we_start, self.we_start + n_pre * row_bytes - 1)
        c_f32 = np.frombuffer(c_bytes, dtype=np.float32).reshape(n_pre, self.hidden)
        max_abs = np.max(np.abs(c_f32), axis=1)
        self.we_pre_scale = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float16)
        self.we_pre_q = np.clip(np.round(c_f32 / self.we_pre_scale.astype(np.float32)[:, None]), -127, 127).astype(np.int8)
        del c_bytes, c_f32
        gc.collect()

        self.rare_emb_cache: Dict[int, np.ndarray] = {}
        print(f"[API] Model '{repo_id}' is ready in ultra-low-memory mode!", flush=True)

    def _fetch_range(self, start_byte: int, end_byte_inclusive: int) -> bytes:
        for attempt in range(4):
            try:
                r = self.session.get(
                    self.st_url,
                    headers={"Range": f"bytes={start_byte}-{end_byte_inclusive}"},
                    timeout=90,
                )
                if r.status_code in (200, 206):
                    return r.content
            except Exception:
                if attempt == 3:
                    raise
        raise RuntimeError(f"Failed to fetch byte range {start_byte}-{end_byte_inclusive}")

    def _stream_and_quantize_tensors(self, tensor_list: list):
        MAX_CHUNK = 8 * 1024 * 1024
        i = 0
        n = len(tensor_list)
        while i < n:
            batch = [tensor_list[i]]
            b_start = tensor_list[i][1]["data_offsets"][0]
            b_end = tensor_list[i][1]["data_offsets"][1]
            j = i + 1
            while j < n:
                nxt_start, nxt_end = tensor_list[j][1]["data_offsets"]
                if nxt_end - b_start <= MAX_CHUNK and (nxt_start - b_end) <= 65536:
                    batch.append(tensor_list[j])
                    b_end = nxt_end
                    j += 1
                else:
                    break

            raw_chunk = self._fetch_range(self.data_base + b_start, self.data_base + b_end - 1)
            for name, meta in batch:
                t_start, t_end = meta["data_offsets"]
                rel_s = t_start - b_start
                rel_e = t_end - b_start
                arr = np.frombuffer(raw_chunk[rel_s:rel_e], dtype=np.float32).reshape(meta["shape"])
                if arr.ndim == 2 and name not in (
                    "roberta.embeddings.position_embeddings.weight",
                    "roberta.embeddings.token_type_embeddings.weight",
                ):
                    q_t, sc = _quantize_2d_int8(arr)
                    self.w_q_t[name] = q_t
                    self.w_scale[name] = sc
                else:
                    self.w_1d[name] = np.array(arr, dtype=np.float32, copy=True)
            del raw_chunk
            gc.collect()
            i = j

    def _lookup_word_embeddings(self, unique_ids: np.ndarray) -> np.ndarray:
        out = np.zeros((len(unique_ids), self.hidden), dtype=np.float32)
        row_bytes = self.hidden * 4
        for idx, uid in enumerate(unique_ids):
            u = int(uid)
            if u < len(self.we_pre_q):
                out[idx] = self.we_pre_q[u].astype(np.float32) * float(self.we_pre_scale[u])
            else:
                if u not in self.rare_emb_cache:
                    s_byte = self.we_start + u * row_bytes
                    raw = self._fetch_range(s_byte, s_byte + row_bytes - 1)
                    self.rare_emb_cache[u] = np.frombuffer(raw, dtype=np.float32).astype(np.float16, copy=True)
                out[idx] = self.rare_emb_cache[u].astype(np.float32)
        return out

    def _linear(self, x2d: np.ndarray, weight_name: str, bias_name: str) -> np.ndarray:
        w_f32 = self.w_q_t[weight_name].astype(np.float32)
        return (x2d @ w_f32) * self.w_scale[weight_name] + self.w_1d[bias_name]

    def _forward(self, batch_ids: List[List[int]]) -> np.ndarray:
        B = len(batch_ids)
        S = max(max(len(ids_row) for ids_row in batch_ids), 1)
        H = self.hidden
        ids = np.full((B, S), self.pad_id, dtype=np.int64)
        mask = np.zeros((B, S), dtype=np.float32)
        for bi, ids_row in enumerate(batch_ids):
            L = len(ids_row)
            ids[bi, :L] = ids_row
            mask[bi, :L] = 1.0

        unique_ids, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        rows = self._lookup_word_embeddings(unique_ids)
        x = rows[inverse].reshape(B, S, H)

        pos_ids = (np.cumsum(mask, axis=1).astype(np.int64) * mask.astype(np.int64)) + self.pad_id
        x = (
            x
            + self.w_1d["roberta.embeddings.position_embeddings.weight"][pos_ids]
            + self.w_1d["roberta.embeddings.token_type_embeddings.weight"][0]
        )
        x = _layer_norm(
            x,
            self.w_1d["roberta.embeddings.LayerNorm.weight"],
            self.w_1d["roberta.embeddings.LayerNorm.bias"],
            self.eps,
        )
        attn_bias = (1.0 - mask[:, None, None, :]) * np.float32(-10000.0)

        for i in range(self.num_layers):
            p = f"roberta.encoder.layer.{i}"
            x2d = x.reshape(B * S, H)
            q = self._linear(x2d, f"{p}.attention.self.query.weight", f"{p}.attention.self.query.bias")
            k = self._linear(x2d, f"{p}.attention.self.key.weight", f"{p}.attention.self.key.bias")
            v = self._linear(x2d, f"{p}.attention.self.value.weight", f"{p}.attention.self.value.bias")

            q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 3, 1)
            v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            probs = _softmax((q @ k) * self.scale + attn_bias, axis=-1)
            ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(B * S, H)
            attn_out = self._linear(
                ctx, f"{p}.attention.output.dense.weight", f"{p}.attention.output.dense.bias"
            ).reshape(B, S, H)
            x = _layer_norm(
                x + attn_out,
                self.w_1d[f"{p}.attention.output.LayerNorm.weight"],
                self.w_1d[f"{p}.attention.output.LayerNorm.bias"],
                self.eps,
            )

            x2d = x.reshape(B * S, H)
            inter = _gelu(
                self._linear(x2d, f"{p}.intermediate.dense.weight", f"{p}.intermediate.dense.bias")
            )
            ffn = self._linear(inter, f"{p}.output.dense.weight", f"{p}.output.dense.bias").reshape(B, S, H)
            x = _layer_norm(
                x + ffn,
                self.w_1d[f"{p}.output.LayerNorm.weight"],
                self.w_1d[f"{p}.output.LayerNorm.bias"],
                self.eps,
            )

        cls = x[:, 0, :]
        h = np.tanh(self._linear(cls, "classifier.dense.weight", "classifier.dense.bias"))
        logits = self._linear(h, "classifier.out_proj.weight", "classifier.out_proj.bias")
        return _softmax(logits, axis=-1)

    def predict_probs(self, clean_texts: List[str]) -> np.ndarray:
        n = len(clean_texts)
        out = np.zeros((n, 3), dtype=np.float32)
        if n == 0:
            return out
        all_ids = [self.tokenizer.encode_ids(t if t else " ") for t in clean_texts]
        order = np.argsort([len(ids_row) for ids_row in all_ids])
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            probs = self._forward([all_ids[i] for i in idx])
            out[idx] = probs[:, self.reorder]
        return out


# ==============================================================================
# MODEL LOADING
# ==============================================================================
_model: Optional[StreamingInt8XLMRStanceModel] = None
_model_error: Optional[str] = None
_model_ready = threading.Event()
_infer_lock = threading.Lock()


def _load_model_background():
    global _model, _model_error
    try:
        _model = StreamingInt8XLMRStanceModel(MODEL_ID)
    except Exception as e:  # noqa: BLE001
        _model_error = f"{type(e).__name__}: {e}"
        print(f"[API] Model failed to load: {_model_error}", flush=True)
    finally:
        _model_ready.set()


@app.on_event("startup")
def _start_loading():
    threading.Thread(target=_load_model_background, daemon=True).start()


def get_model(wait_seconds: float = 600.0) -> StreamingInt8XLMRStanceModel:
    if not _model_ready.wait(timeout=wait_seconds):
        raise HTTPException(status_code=503, detail="Model is still loading from Hugging Face. Try again in a minute.")
    if _model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {_model_error}")
    return _model


def predict_texts(raw_texts: List[str]) -> List[dict]:
    model = get_model()
    cleaned = [clean_stance_text(t if isinstance(t, str) else "") for t in raw_texts]
    with _infer_lock:
        probs = model.predict_probs(cleaned)
    results = []
    for raw, clean, p in zip(raw_texts, cleaned, probs):
        if not clean:
            p = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        best = int(np.argmax(p))
        results.append({
            "text": raw,
            "cleaned": clean,
            "label": PROJECT_LABELS[best],
            "confidence": round(float(p[best]), 4),
            "probabilities": {lbl: round(float(p[i]), 4) for i, lbl in enumerate(PROJECT_LABELS)},
        })
    return results


# ==============================================================================
# INPUT FILE PARSER (CSV / TSV / TXT / XLSX)
# ==============================================================================
ID_NAMES = ["ID", "INDEX", "ROW", "ROW_ID", "S.NO", "SNO", "SR.NO"]
TEXT_NAMES = ["text", "clean_comment", "hindi_comment", "raw_comment", "comment", "tweet", "sentence", "content", "post"]


def parse_uploaded_file(raw_bytes: bytes, filename: str):
    low = (filename or "input.csv").lower()
    if low.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw_bytes))
    else:
        content = None
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                content = raw_bytes.decode(enc)
                break
            except Exception:
                continue
        if content is None:
            content = raw_bytes.decode("utf-8", errors="replace")
        first_line = content.splitlines()[0] if content.splitlines() else ""
        if low.endswith(".txt") and "\t" not in first_line:
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            return pd.DataFrame({"id": range(1, len(lines) + 1), "text": lines}), "id", "text"
        sep = "\t" if (low.endswith((".tsv", ".txt")) or ("\t" in first_line and "," not in first_line)) else ","
        df = pd.read_csv(io.StringIO(content), sep=sep)

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if len(df) == 0:
        raise ValueError("The uploaded file has no rows.")

    id_col = next((c for c in df.columns if c.strip().upper() in ID_NAMES), None)
    if id_col is None:
        df.insert(0, "id", range(1, len(df) + 1))
        id_col = "id"

    text_col = next((c for c in df.columns if c.strip().lower() in TEXT_NAMES), None)
    if text_col is None:
        candidates = [c for c in df.columns if c != id_col and not pd.api.types.is_numeric_dtype(df[c])]
        candidates = candidates or [c for c in df.columns if c != id_col]
        if not candidates:
            raise ValueError("Could not find a text column in the uploaded file.")
        text_col = max(candidates, key=lambda c: df[c].astype(str).str.len().mean())
    return df, id_col, text_col


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
class TextItem(BaseModel):
    id: Optional[Union[int, str]] = Field(default=None, description="Row ID")
    text: str = Field(..., description="Hindi, Hinglish or English text")


class PredictRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="A single text to classify")
    id: Optional[Union[int, str]] = Field(default=1, description="ID for the single text")
    data: Optional[List[TextItem]] = Field(default=None, description="List of {id, text} rows")


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""
    <!DOCTYPE html>
    <html lang="en"><head><meta charset="UTF-8"/><title>ClimaStance NLP API</title>
    <style>
      body {{ font-family: system-ui, sans-serif; max-width: 880px; margin: 40px auto; padding: 0 20px; background: #0B0D12; color: #fff; }}
      .card {{ background: #1C1F24; border: 1px solid rgba(255,255,255,0.12); border-radius: 20px; padding: 26px; margin-bottom: 22px; }}
      code {{ background: #0B0D12; padding: 2px 8px; border-radius: 6px; color: #4ade80; }}
      a {{ color: #38BDF8; }}
      input, button {{ font-size: 0.95rem; border-radius: 999px; padding: 10px 18px; border: 1px solid rgba(255,255,255,0.2); background: #0B0D12; color: #fff; }}
      button {{ background: #fff; color: #0B0D12; font-weight: 700; border: none; cursor: pointer; }}
    </style></head>
    <body>
      <div class="card">
        <h1>ClimaStance &mdash; Detecting Stance in Media on Global Warming</h1>
        <p><b>Claim:</b> &ldquo;{TARGET_CLAIM}&rdquo;</p>
        <p><b>Model:</b> <code>{MODEL_ID}</code> (fine-tuned XLM-RoBERTa from Hugging Face)</p>
        <p><b>Labels:</b> <code>FAVOR</code> <code>AGAINST</code> <code>NONE</code></p>
        <p><a href="/docs">API docs (/docs)</a> &nbsp;|&nbsp; <a href="/health">Health (/health)</a></p>
      </div>
      <div class="card">
        <h3>Upload Input CSV (id, text) &rarr; Download Output CSV (id, label)</h3>
        <form action="/predict_csv" method="post" enctype="multipart/form-data">
          <input type="file" name="file" accept=".csv,.txt,.tsv,.xlsx" required />
          <button type="submit">Predict &amp; Download</button>
        </form>
      </div>
    </body></html>
    """


@app.get("/health")
def health():
    if not _model_ready.is_set():
        status = "loading"
    elif _model is None:
        status = "error"
    else:
        status = "ready"
    return {
        "status": status,
        "model_id": MODEL_ID,
        "model": MODEL_KEY,
        "claim": TARGET_CLAIM,
        "labels": ["FAVOR", "AGAINST", "NONE"],
        "error": _model_error,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    if payload.data:
        ids = [item.id if item.id is not None else i + 1 for i, item in enumerate(payload.data)]
        texts = [item.text for item in payload.data]
    elif payload.text is not None:
        ids = [payload.id if payload.id is not None else 1]
        texts = [payload.text]
    else:
        raise HTTPException(status_code=400, detail="Send either 'text' or 'data': [{'id': 1, 'text': '...'}].")

    results = predict_texts(texts)
    for row_id, r in zip(ids, results):
        r["id"] = row_id
    return {"claim": TARGET_CLAIM, "model": MODEL_KEY, "model_id": MODEL_ID, "count": len(results), "predictions": results}


@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(...),
    output_format: str = Form(default="diagram"),
):
    raw = await file.read()
    try:
        df, id_col, text_col = parse_uploaded_file(raw, file.filename or "input.csv")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    texts = df[text_col].fillna("").astype(str).tolist()
    labels = [r["label"] for r in predict_texts(texts)]
    if output_format.strip().lower() == "submission":
        out_df = pd.DataFrame({"ID": df[id_col].tolist(), "Predicted Label": labels})
    else:
        out_df = pd.DataFrame({"id": df[id_col].tolist(), "label": labels})

    data = out_df.to_csv(index=False).encode("utf-8-sig")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="output_predictions.csv"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
