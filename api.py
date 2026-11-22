"""
Render FastAPI Backend for NLP Stance Detection
===============================================
Project : Detecting Stance in Media on Global Warming for Indic Languages (CSET 346)
Claim   : "Climate change and global warming is a serious concern."
Model   : Fine-tuned XLM-RoBERTa, downloaded from Hugging Face (richa3137/climate-stance-xlmr)
Labels  : FAVOR, AGAINST, NONE

Professor's workflow:
  Step 1: Project is hosted on Render (uvicorn api:app --host 0.0.0.0 --port $PORT)
  Step 2: Render exposes this REST API (/predict, /predict_csv, /health, /docs)
  Step 3: Client sends Input CSV (id | text)
  Step 4: The fine-tuned model predicts the stance label
  Step 5: Output CSV (id | label) is returned

Memory note:
  The model file is ~1.1 GB but Render's free plan has 512 MB RAM. So the weights are NOT
  loaded into RAM all at once. They stay on disk and are read one layer at a time for each
  request (safetensors.safe_open). This keeps RAM low, but each prediction is slower.
"""
import html
import io
import json
import os
import re
import threading
import unicodedata
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from safetensors import safe_open
from scipy.special import erf
from tokenizers import Tokenizer

# ==============================================================================
# CONFIGURATION
# ==============================================================================
APP_TITLE = "ClimaStance NLP REST API - Detecting Stance in Media on Global Warming"
TARGET_CLAIM = "Climate change and global warming is a serious concern."
MODEL_ID = os.environ.get("MODEL_ID", "richa3137/climate-stance-xlmr")
MODEL_KEY = "xlmr"
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "128"))
CACHE_DIR = os.environ.get("HF_HOME", "/tmp/hf_cache")
BATCH_SIZE = int(os.environ.get("INFER_BATCH_SIZE", "16"))

# Hugging Face model uses "Against / None / Favour"; the project uses AGAINST / NONE / FAVOR.
HF_TO_PROJECT_LABEL = {"against": "AGAINST", "none": "NONE", "favour": "FAVOR", "favor": "FAVOR"}
PROJECT_LABELS = ["AGAINST", "NONE", "FAVOR"]

app = FastAPI(
    title=APP_TITLE,
    description=(
        f"REST API for climate stance detection on the claim: '{TARGET_CLAIM}'. "
        f"Model: {MODEL_ID} (fine-tuned XLM-RoBERTa, loaded from Hugging Face). "
        "Send JSON to /predict or an Input CSV (id, text) to /predict_csv to get an Output CSV (id, label)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# TEXT CLEANING (same function used when the Hugging Face model was trained)
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


# ==============================================================================
# LOW-MEMORY XLM-ROBERTA INFERENCE ENGINE (weights read from disk layer by layer)
# ==============================================================================
def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: np.float32) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def _gelu(x: np.ndarray) -> np.ndarray:
    return np.float32(0.5) * x * (np.float32(1.0) + erf(x / np.float32(np.sqrt(2.0))).astype(np.float32))


class HuggingFaceXLMRStanceModel:
    def __init__(self, repo_id: str):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[API] Downloading '{repo_id}' from Hugging Face (first start can take 1-2 minutes)...", flush=True)
        cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir=CACHE_DIR)
        tok_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", cache_dir=CACHE_DIR)
        self.st_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", cache_dir=CACHE_DIR)

        with open(cfg_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.hidden = int(self.config["hidden_size"])
        self.head_dim = self.hidden // self.num_heads
        self.eps = np.float32(self.config.get("layer_norm_eps", 1e-5))
        self.pad_id = int(self.config.get("pad_token_id", 1))
        self.scale = np.float32(1.0 / np.sqrt(self.head_dim))

        # Map model output index -> project label (AGAINST / NONE / FAVOR)
        id2label = {int(k): v for k, v in self.config["id2label"].items()}
        self.out_labels = [HF_TO_PROJECT_LABEL.get(str(id2label[i]).strip().lower(), str(id2label[i]).upper())
                           for i in range(len(id2label))]
        # Column order to convert model output -> [AGAINST, NONE, FAVOR]
        self.reorder = [self.out_labels.index(lbl) for lbl in PROJECT_LABELS]

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LENGTH)

        # Keep only the small tensors in RAM (position embeddings, embedding LayerNorm)
        with safe_open(self.st_path, framework="np") as f:
            self.pos_emb = f.get_tensor("roberta.embeddings.position_embeddings.weight").astype(np.float32)
            self.type_emb = f.get_tensor("roberta.embeddings.token_type_embeddings.weight").astype(np.float32)[0]
            self.emb_ln_w = f.get_tensor("roberta.embeddings.LayerNorm.weight").astype(np.float32)
            self.emb_ln_b = f.get_tensor("roberta.embeddings.LayerNorm.bias").astype(np.float32)
        print(f"[API] Model '{repo_id}' is ready.", flush=True)

    def _forward(self, f, batch_enc) -> np.ndarray:
        B = len(batch_enc)
        S = max(max(len(e.ids) for e in batch_enc), 1)
        H = self.hidden
        ids = np.full((B, S), self.pad_id, dtype=np.int64)
        mask = np.zeros((B, S), dtype=np.float32)
        for bi, e in enumerate(batch_enc):
            L = len(e.ids)
            ids[bi, :L] = e.ids
            mask[bi, :L] = 1.0

        # Word embeddings: read only the rows that are needed (not the whole 768 MB table)
        word_slice = f.get_slice("roberta.embeddings.word_embeddings.weight")
        unique_ids, inverse = np.unique(ids.reshape(-1), return_inverse=True)
        rows = np.stack([word_slice[int(u):int(u) + 1][0] for u in unique_ids], axis=0).astype(np.float32)
        x = rows[inverse].reshape(B, S, H)

        pos_ids = (np.cumsum(mask, axis=1).astype(np.int64) * mask.astype(np.int64)) + self.pad_id
        x = x + self.pos_emb[pos_ids] + self.type_emb
        x = _layer_norm(x, self.emb_ln_w, self.emb_ln_b, self.eps)
        attn_bias = (1.0 - mask[:, None, None, :]) * np.float32(-10000.0)

        def T(name):
            return f.get_tensor(name).astype(np.float32, copy=False)

        for i in range(self.num_layers):
            p = f"roberta.encoder.layer.{i}"
            x2d = x.reshape(B * S, H)
            q = (x2d @ T(f"{p}.attention.self.query.weight").T + T(f"{p}.attention.self.query.bias"))
            k = (x2d @ T(f"{p}.attention.self.key.weight").T + T(f"{p}.attention.self.key.bias"))
            v = (x2d @ T(f"{p}.attention.self.value.weight").T + T(f"{p}.attention.self.value.bias"))
            q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 3, 1)
            v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

            probs = _softmax((q @ k) * self.scale + attn_bias, axis=-1)
            ctx = (probs @ v).transpose(0, 2, 1, 3).reshape(B * S, H)
            attn_out = (ctx @ T(f"{p}.attention.output.dense.weight").T
                        + T(f"{p}.attention.output.dense.bias")).reshape(B, S, H)
            x = _layer_norm(x + attn_out, T(f"{p}.attention.output.LayerNorm.weight"),
                            T(f"{p}.attention.output.LayerNorm.bias"), self.eps)

            x2d = x.reshape(B * S, H)
            inter = _gelu(x2d @ T(f"{p}.intermediate.dense.weight").T + T(f"{p}.intermediate.dense.bias"))
            ffn = (inter @ T(f"{p}.output.dense.weight").T + T(f"{p}.output.dense.bias")).reshape(B, S, H)
            x = _layer_norm(x + ffn, T(f"{p}.output.LayerNorm.weight"),
                            T(f"{p}.output.LayerNorm.bias"), self.eps)

        cls = x[:, 0, :]
        h = np.tanh(cls @ T("classifier.dense.weight").T + T("classifier.dense.bias"))
        logits = h @ T("classifier.out_proj.weight").T + T("classifier.out_proj.bias")
        return _softmax(logits, axis=-1)

    def predict_probs(self, clean_texts: List[str]) -> np.ndarray:
        """Returns probabilities in the column order [AGAINST, NONE, FAVOR]."""
        n = len(clean_texts)
        out = np.zeros((n, 3), dtype=np.float32)
        if n == 0:
            return out
        encodings = self.tokenizer.encode_batch([t if t else " " for t in clean_texts])
        order = np.argsort([len(e.ids) for e in encodings])
        with safe_open(self.st_path, framework="np") as f:
            for start in range(0, n, BATCH_SIZE):
                idx = order[start:start + BATCH_SIZE]
                probs = self._forward(f, [encodings[i] for i in idx])
                out[idx] = probs[:, self.reorder]
        return out


# ==============================================================================
# MODEL LOADING (in the background, so Render sees the server as "up" immediately)
# ==============================================================================
_model: Optional[HuggingFaceXLMRStanceModel] = None
_model_error: Optional[str] = None
_model_ready = threading.Event()
_infer_lock = threading.Lock()


def _load_model_background():
    global _model, _model_error
    try:
        _model = HuggingFaceXLMRStanceModel(MODEL_ID)
    except Exception as e:  # noqa: BLE001
        _model_error = f"{type(e).__name__}: {e}"
        print(f"[API] Model failed to load: {_model_error}", flush=True)
    finally:
        _model_ready.set()


@app.on_event("startup")
def _start_loading():
    threading.Thread(target=_load_model_background, daemon=True).start()


def get_model(wait_seconds: float = 600.0) -> HuggingFaceXLMRStanceModel:
    if not _model_ready.wait(timeout=wait_seconds):
        raise HTTPException(status_code=503, detail="Model is still downloading from Hugging Face. Try again in a minute.")
    if _model is None:
        raise HTTPException(status_code=500, detail=f"Model failed to load: {_model_error}")
    return _model


def predict_texts(raw_texts: List[str]) -> List[dict]:
    model = get_model()
    cleaned = [clean_stance_text(t if isinstance(t, str) else "") for t in raw_texts]
    with _infer_lock:  # one request at a time, so RAM stays under Render's 512 MB limit
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
    """
    Upload an Input CSV (id, text). Returns an Output CSV:
      output_format='diagram'    -> id,label              (professor's diagram)
      output_format='submission' -> ID,Predicted Label    (project submission format)
    """
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
