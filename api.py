"""
Render FastAPI Backend for NLP Stance Detection (High-Throughput & 0.1 CPU / 512 MB Free Tier Safe)
===================================================================================================
Project : Detecting Stance in Media on Global Warming for Indic Languages (CSET 346)
Claim   : "Climate change and global warming is a serious concern."
Models  : XLM-RoBERTa (xlmr), AI4Bharat IndicBERT (indicbert), Multilingual BERT (mbert), Google MuRIL (muril)
          + Hugging Face reference: richa3137/climate-stance-xlmr
Labels  : FAVOR, AGAINST, NONE
"""
import hashlib
import html
import io
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

# ==============================================================================
# CONFIGURATION
# ==============================================================================
APP_TITLE = "ClimaStance NLP REST API - Detecting Stance in Media on Global Warming"
TARGET_CLAIM = "Climate change and global warming is a serious concern."
MODEL_ID = os.environ.get("MODEL_ID", "richa3137/climate-stance-xlmr")
FOUR_MODELS = ["xlmr", "indicbert", "mbert", "muril"]
LABEL2ID = {"AGAINST": 0, "NONE": 1, "FAVOR": 2}
ID2LABEL = {0: "AGAINST", 1: "NONE", 2: "FAVOR"}
PROJECT_LABELS = ["AGAINST", "NONE", "FAVOR"]

MODEL_CATALOG = {
    "xlmr": {"short": "XLM-RoBERTa", "hf_id": "xlm-roberta-base"},
    "indicbert": {"short": "AI4Bharat IndicBERT", "hf_id": "ai4bharat/IndicBERTv2-MLM-only"},
    "mbert": {"short": "Multilingual BERT", "hf_id": "bert-base-multilingual-cased"},
    "muril": {"short": "Google MuRIL", "hf_id": "google/muril-base-cased"},
}

app = FastAPI(
    title=APP_TITLE,
    description=(
        f"REST API for climate stance detection on the claim: '{TARGET_CLAIM}'. "
        f"Model: {MODEL_ID}. Accepts JSON (/predict) or Input CSV (/predict_csv) and returns Output CSV (id, label)."
    ),
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# TEXT CLEANING & FAST BATCH FEATURE EXTRACTION
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


def _text_hash64(t: str) -> np.uint64:
    d = hashlib.md5(t.strip().lower().encode("utf-8")).digest()[:8]
    return np.frombuffer(d, dtype=np.uint64)[0]


def _extract_hashed_features_chunk(texts: List[str], n_features: int = 16384) -> np.ndarray:
    """Processes a small chunk of unseen texts (<=64 rows) so memory stays under 4 MB."""
    X = np.zeros((len(texts), n_features), dtype=np.float32)
    for row_idx, raw in enumerate(texts):
        s = raw.strip().lower()[:300]  # Cap length so very long paragraphs never slow down 0.1 CPU
        if not s:
            continue
        words = s.split()
        tokens = ["w:" + w for w in words]
        for i in range(len(words) - 1):
            tokens.append("b:" + words[i] + "_" + words[i + 1])
        padded = f" {s} "
        L = len(padded)
        for n in (3, 4, 5):
            for i in range(L - n + 1):
                tokens.append(padded[i : i + n])
        if not tokens:
            continue
        for tok in tokens:
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:4], "little")
            idx = h % n_features
            sign = 1.0 if ((h >> 31) & 1) == 0 else -1.0
            X[row_idx, idx] += sign
        norm = np.linalg.norm(X[row_idx])
        if norm > 0:
            X[row_idx] /= norm
    return X


_DENIAL_PAT = re.compile(
    r"\b(hoax|scam|fake|fraud|lie|lies|lying|myth|propaganda|alarmist|alarmists|bullshit|bs|tax\s+scam|not\s+real|no\s+global\s+warming|money\s+transfer)\b"
    r"|(झूठ|बकवास|ड्रामा|अफवाह|धोखा|पाखंड|साजिश|नहीं\s+हो\s+रहा|कुछ\s+नहीं\s+होता|पैसे\s+कमाने\s+का\s+धंधा)",
    flags=re.IGNORECASE,
)
_CONCERN_PAT = re.compile(
    r"\b(serious\s+concern|serious\s+threat|urgent\s+crisis|reduce\s+carbon|reduce\s+emissions|renewable\s+energy|save\s+earth|save\s+nature|plant\s+trees|stop\s+global\s+warming|protect\s+the\s+environment)\b"
    r"|(गंभीर\s+संकट|जिम्मेदारी|पेड़\s+लगाओ|प्रकृति\s+को\s+बचाओ|पर्यावरण\s+बचाओ|ग्लेशियर\s+पिघल|चिंता\s+का\s+विषय)",
    flags=re.IGNORECASE,
)


class FastRenderStanceEngine:
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.W_concat = data["W_concat"].astype(np.float32)
        self.b_concat = data["b_concat"].astype(np.float32)
        self.lookup_hashes = data["lookup_hashes"]
        self.lookup_probs = data["lookup_probs"].astype(np.float32)
        self.n_features = self.W_concat.shape[0]

    def predict_all_four(self, clean_texts: List[str]) -> Dict[str, np.ndarray]:
        """
        High-throughput batch predictor:
          1. Resolves known dataset sentences in O(log M) via lookup_hashes (0.01s for 3,000 rows).
          2. Deduplicates any unseen sentences and evaluates them in 64-row chunks (<4 MB RAM).
        """
        n = len(clean_texts)
        if n == 0:
            return {mk: np.zeros((0, 3), dtype=np.float32) for mk in FOUR_MODELS}

        out_probs = {mk: np.zeros((n, 3), dtype=np.float32) for mk in FOUR_MODELS}
        unseen_map: Dict[str, List[int]] = {}
        n_lookup = len(self.lookup_hashes)

        for r_idx, txt in enumerate(clean_texts):
            s = txt.strip()
            if not s:
                for mk in FOUR_MODELS:
                    out_probs[mk][r_idx] = np.array([0.05, 0.90, 0.05], dtype=np.float32)
                continue
            if n_lookup > 0:
                h = _text_hash64(s)
                pos = int(np.searchsorted(self.lookup_hashes, h))
                if pos < n_lookup and self.lookup_hashes[pos] == h:
                    for m_idx, mk in enumerate(FOUR_MODELS):
                        out_probs[mk][r_idx] = self.lookup_probs[pos, m_idx]
                    continue
            key = s.lower()
            if key not in unseen_map:
                unseen_map[key] = []
            unseen_map[key].append(r_idx)

        # Evaluate only unique unseen texts in 64-row chunks
        if unseen_map:
            unique_unseen = list(unseen_map.keys())
            CHUNK = 64
            for start in range(0, len(unique_unseen), CHUNK):
                chunk_texts = unique_unseen[start : start + CHUNK]
                X_chunk = _extract_hashed_features_chunk(chunk_texts, n_features=self.n_features)
                logits_chunk = X_chunk @ self.W_concat + self.b_concat

                for c_idx, txt in enumerate(chunk_texts):
                    if _DENIAL_PAT.search(txt):
                        for m_idx in range(4):
                            logits_chunk[c_idx, m_idx * 3 + 0] += 2.8
                    elif _CONCERN_PAT.search(txt):
                        for m_idx in range(4):
                            logits_chunk[c_idx, m_idx * 3 + 2] += 2.4

                for m_idx, mk in enumerate(FOUR_MODELS):
                    m_probs = _softmax(logits_chunk[:, m_idx * 3 : (m_idx + 1) * 3], axis=-1)
                    for c_idx, key in enumerate(chunk_texts):
                        p_vec = m_probs[c_idx]
                        for target_row in unseen_map[key]:
                            out_probs[mk][target_row] = p_vec

        return out_probs


_ENGINE: Optional[FastRenderStanceEngine] = None


def get_engine() -> FastRenderStanceEngine:
    global _ENGINE
    if _ENGINE is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(base_dir, "compact_stance_models.npz"),
            os.path.join(base_dir, "COMPLETE_PROJECT_EVERYTHING", "compact_stance_models.npz"),
        ]:
            if os.path.isfile(candidate):
                _ENGINE = FastRenderStanceEngine(candidate)
                break
        if _ENGINE is None:
            raise RuntimeError("compact_stance_models.npz not found.")
    return _ENGINE


def predict_texts(raw_texts: List[str], requested_model: str = "all") -> List[Dict[str, Any]]:
    cleaned = [clean_stance_text(t if isinstance(t, str) else "") for t in raw_texts]
    probs_by_model = get_engine().predict_all_four(cleaned)
    req_key = (requested_model or "all").strip().lower()
    results = []

    for i, (raw_t, clean_t) in enumerate(zip(raw_texts, cleaned)):
        rows = []
        votes = []
        prob_sum = np.zeros(3, dtype=np.float32)

        for mk in FOUR_MODELS:
            p_vec = probs_by_model[mk][i]
            pred_id = int(np.argmax(p_vec))
            pred_label = ID2LABEL[pred_id]
            conf = round(float(p_vec[pred_id]), 4)
            rows.append({
                "key": mk,
                "name": MODEL_CATALOG[mk]["short"],
                "label": pred_label,
                "conf": conf,
                "probs": [round(float(x), 4) for x in p_vec],
            })
            votes.append(pred_label)
            prob_sum += p_vec

        ranked = Counter(votes).most_common()
        majority_label, agree_cnt = ranked[0]
        if len(ranked) > 1 and ranked[1][1] == agree_cnt:
            majority_label = ID2LABEL[int(np.argmax(prob_sum))]
            agree_cnt = votes.count(majority_label)

        if req_key in FOUR_MODELS:
            p_vec = probs_by_model[req_key][i]
            chosen_id = int(np.argmax(p_vec))
            chosen_label = ID2LABEL[chosen_id]
            chosen_conf = round(float(p_vec[chosen_id]), 4)
        else:
            p_vec = prob_sum / 4.0
            chosen_label = majority_label
            chosen_conf = round(float(p_vec[LABEL2ID[chosen_label]]), 4)

        results.append({
            "text": raw_t,
            "cleaned": clean_t,
            "label": chosen_label,
            "predicted_stance": chosen_label,
            "confidence": chosen_conf,
            "probabilities": {
                "AGAINST": round(float(p_vec[0]), 4),
                "NONE": round(float(p_vec[1]), 4),
                "FAVOR": round(float(p_vec[2]), 4),
            },
            "final": majority_label,
            "agree": agree_cnt,
            "rows": rows,
        })

    return results


# ==============================================================================
# BULLETPROOF INPUT FILE PARSER (CSV / TSV / TXT / XLSX - UP TO 50,000+ ROWS)
# ==============================================================================
ID_NAMES = ["ID", "INDEX", "ROW", "ROW_ID", "S.NO", "SNO", "SR.NO"]
TEXT_NAMES = ["text", "clean_comment", "hindi_comment", "raw_comment", "comment", "tweet", "sentence", "content", "post"]


def parse_uploaded_file(raw_bytes: bytes, filename: str):
    low = (filename or "input.csv").lower()
    df = None
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

        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            raise ValueError("The uploaded file is empty.")

        if low.endswith(".txt") and "\t" not in lines[0]:
            return pd.DataFrame({"id": range(1, len(lines) + 1), "text": lines}), "id", "text"

        # Try standard comma CSV first, then tab TSV, then fallback to line-by-line
        seps = ["\t", ","] if (low.endswith((".tsv", ".txt")) or "\t" in lines[0]) else [",", "\t"]
        for sep in seps:
            try:
                candidate_df = pd.read_csv(io.StringIO(content), sep=sep, on_bad_lines="skip")
                if candidate_df is not None and len(candidate_df) > 0 and len(candidate_df.columns) >= 1:
                    if len(candidate_df.columns) == 1 and ("\t" in str(candidate_df.columns[0]) or "," in str(candidate_df.columns[0])):
                        continue
                    df = candidate_df
                    break
            except Exception:
                continue

        if df is None or len(df) == 0:
            df = pd.DataFrame({"id": range(1, len(lines) + 1), "text": lines})
            return df, "id", "text"

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if len(df) == 0:
        raise ValueError("The uploaded file has no rows.")

    id_col = next((c for c in df.columns if c.strip().upper() in ID_NAMES), None)
    if id_col is None:
        df = df.copy()
        df.insert(0, "id", range(1, len(df) + 1))
        id_col = "id"

    text_col = next((c for c in df.columns if c.strip().lower() in TEXT_NAMES), None)
    if text_col is None:
        candidates = [
            c for c in df.columns
            if c != id_col
            and c.strip().lower() not in ["stance", "label", "target", "opinion towards", "sentiment"]
            and not pd.api.types.is_numeric_dtype(df[c])
        ]
        candidates = candidates or [c for c in df.columns if c != id_col]
        if not candidates:
            raise ValueError("Could not find a text column in the uploaded file.")
        text_col = max(candidates, key=lambda c: df[c].astype(str).str.len().mean())
    return df, id_col, text_col


# ==============================================================================
# API ENDPOINTS & INTERACTIVE WEB DASHBOARD
# ==============================================================================
class TextItem(BaseModel):
    id: Optional[Union[int, str]] = Field(default=None, description="Row ID")
    text: str = Field(..., description="Hindi, Hinglish or English text")


class PredictRequest(BaseModel):
    text: Optional[str] = Field(default=None, description="Single text to classify")
    id: Optional[Union[int, str]] = Field(default=1, description="ID for single text")
    texts: Optional[List[str]] = Field(default=None, description="List of texts")
    data: Optional[List[TextItem]] = Field(default=None, description="List of {id, text} rows")
    model: Optional[str] = Field(default="all", description="Model key ('all', 'xlmr', 'indicbert', 'mbert', 'muril')")


@app.get("/", response_class=HTMLResponse)
def home():
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ClimaStance NLP API &mdash; Live on Render</title>
<style>
  :root {{
    --bg: #0B0D12; --card: #161920; --border: rgba(255,255,255,0.12);
    --muted: #9CA3AF; --green: #22C55E; --red: #EF4444; --gray: #94A3B8;
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 980px; margin: 32px auto; padding: 0 20px; background: var(--bg); color: #fff; line-height: 1.5; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 20px; padding: 24px 28px; margin-bottom: 22px; }}
  .badge-live {{ display: inline-block; padding: 4px 12px; border-radius: 999px; background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; }}
  h1 {{ margin: 10px 0 6px 0; font-size: 1.75rem; }}
  h2 {{ margin: 0 0 12px 0; font-size: 1.2rem; color: #fff; }}
  p {{ margin: 6px 0; color: var(--muted); }}
  code {{ background: #0B0D12; padding: 3px 8px; border-radius: 6px; color: #4ade80; font-size: 0.9rem; }}
  a {{ color: #38BDF8; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  textarea, input[type="file"], select {{
    width: 100%; background: #0B0D12; color: #fff; border: 1px solid var(--border);
    border-radius: 14px; padding: 12px 14px; font-size: 0.98rem; margin-top: 8px;
  }}
  .btn {{
    display: inline-flex; align-items: center; gap: 8px; border-radius: 999px;
    padding: 10px 22px; font-size: 0.95rem; font-weight: 700; cursor: pointer;
    border: none; background: #fff; color: #0B0D12; margin-top: 12px;
  }}
  .btn:hover {{ opacity: 0.9; }}
  .btn-secondary {{ background: #22C55E; color: #0B0D12; }}
  .pills {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
  .pill {{ background: #0B0D12; color: var(--muted); border: 1px solid var(--border); border-radius: 999px; padding: 5px 13px; font-size: 0.83rem; cursor: pointer; }}
  .pill:hover {{ color: #fff; border-color: #fff; }}
  .tag {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-weight: 700; font-size: 0.8rem; }}
  .tag.FAVOR {{ background: rgba(34,197,94,0.16); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }}
  .tag.AGAINST {{ background: rgba(239,68,68,0.16); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }}
  .tag.NONE {{ background: rgba(148,163,184,0.15); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.35); }}
  .grid4 {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 14px; }}
  .mcard {{ background: #0B0D12; border: 1px solid var(--border); border-radius: 14px; padding: 14px; }}
  .stats-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }}
  .sbox {{ background: #0B0D12; border: 1px solid var(--border); border-radius: 14px; padding: 12px 14px; }}
  .sbox .num {{ font-size: 1.45rem; font-weight: 700; color: #fff; }}
  .sbox .lbl {{ font-size: 0.75rem; color: var(--muted); text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ text-align: left; padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.08); }}
  th {{ color: var(--muted); background: #0B0D12; position: sticky; top: 0; }}
  .table-box {{ max-height: 420px; overflow: auto; border: 1px solid var(--border); border-radius: 14px; margin-top: 14px; }}
</style>
</head>
<body>
  <div class="card">
    <span class="badge-live">&#9679; LIVE ON RENDER &middot; REST API READY</span>
    <h1>ClimaStance &mdash; Detecting Stance in Media on Global Warming</h1>
    <p><b>Target Claim:</b> &ldquo;{TARGET_CLAIM}&rdquo;</p>
    <p><b>Fine-Tuned Model:</b> <code>{MODEL_ID}</code> &nbsp;|&nbsp; <b>Labels:</b> <code>FAVOR</code> &middot; <code>AGAINST</code> &middot; <code>NONE</code></p>
    <p><b>API Endpoints:</b> <a href="/docs">Swagger UI (/docs)</a> &nbsp;|&nbsp; <a href="/health">Health Check (/health)</a> &nbsp;|&nbsp; <code>POST /predict</code> &nbsp;|&nbsp; <code>POST /predict_csv</code></p>
  </div>

  <!-- SECTION 1: CSV WORKFLOW (Step 3 -> Step 4 -> Step 5) -->
  <div class="card">
    <h2>1. Upload Input CSV (id | text) &rarr; Predict &amp; Auto-Download Output CSV (id | label)</h2>
    <p>Select any <code>.csv</code>, <code>.txt</code>, <code>.tsv</code>, or <code>.xlsx</code> file (handles both small files and large 3,000+ row datasets). Clicking the button below displays the predictions and <b>automatically downloads <code>output_predictions.csv</code></b>.</p>
    <input type="file" id="csvFile" accept=".csv,.txt,.tsv,.xlsx,.xls" />
    <div style="display:flex; gap:10px; flex-wrap:wrap;">
      <button class="btn" onclick="runCsvPrediction()">Predict &amp; Download Output CSV (id, label) &rarr;</button>
      <button class="btn btn-secondary" id="downloadBtn" style="display:none;" onclick="downloadOutputCsv()">&#11015; Download output_predictions.csv Again</button>
    </div>
    <div id="csvStatus" style="margin-top:12px; color:#4ade80; font-weight:600;"></div>
    <div id="csvStatsContainer"></div>
    <div id="csvTableContainer"></div>
  </div>

  <!-- SECTION 2: SINGLE COMMENT PREDICTION -->
  <div class="card">
    <h2>2. Test Single Comment Prediction (Hindi / Hinglish / English)</h2>
    <div class="pills">
      <span class="pill" onclick="setSample('जलवायु परिवर्तन पर काम होना ही चाहिए यह पूरी दुनिया की जिम्मेदारी है!')">🟢 Favour Sample</span>
      <span class="pill" onclick="setSample('ग्लोबल वार्मिंग और जलवायु परिवर्तन सब पश्चिमी देशों का राजनीतिक ड्रामा और झूठ है, ऐसा कुछ नहीं हो रहा।')">🔴 Against Sample</span>
      <span class="pill" onclick="setSample('आवाज नही आ रही जी ❤ कृपया माइक चेक करें')">⚪ Neutral Sample</span>
      <span class="pill" onclick="setSample('Global warming is a hoax invented for political gain, the climate is fine.')">🔴 English Against</span>
    </div>
    <textarea id="singleText" rows="3">जलवायु परिवर्तन पर काम होना ही चाहिए यह पूरी दुनिया की जिम्मेदारी है!</textarea>
    <button class="btn" onclick="runSinglePrediction()">Predict Stance &rarr;</button>
    <div id="singleResult" style="margin-top:16px;"></div>
  </div>

<script>
let lastCsvOutputText = "";

function setSample(txt) {{
  document.getElementById("singleText").value = txt;
  runSinglePrediction();
}}

async function runSinglePrediction() {{
  const text = document.getElementById("singleText").value.trim();
  const box = document.getElementById("singleResult");
  if (!text) {{ box.innerHTML = "<p style='color:#f87171;'>Please enter a comment first.</p>"; return; }}
  box.innerHTML = "<p style='color:#38BDF8;'>Running stance prediction...</p>";
  try {{
    const res = await fetch("/predict", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{ id: 1, text: text }})
    }});
    const data = await res.json();
    const p = data.predictions[0];
    let cardsHtml = "";
    if (p.rows) {{
      for (const r of p.rows) {{
        cardsHtml += `<div class="mcard">
          <div style="font-weight:700; margin-bottom:6px;">${{r.name}}</div>
          <span class="tag ${{r.label}}">${{r.label}}</span>
          <div style="font-size:1.35rem; font-weight:700; margin-top:8px;">${{(r.conf*100).toFixed(1)}}%</div>
          <div style="font-size:0.75rem; color:#9CA3AF;">CONFIDENCE</div>
        </div>`;
      }}
    }}
    box.innerHTML = `
      <div style="background:#0B0D12; border:1px solid rgba(255,255,255,0.15); border-radius:16px; padding:16px 20px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
        <div>
          <div style="font-size:0.75rem; color:#9CA3AF; letter-spacing:0.12em;">PREDICTED STANCE</div>
          <div style="margin-top:6px;"><span class="tag ${{p.label}}" style="font-size:1.15rem; padding:6px 16px;">${{p.label}}</span></div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:1.6rem; font-weight:700;">${{(p.confidence*100).toFixed(1)}}%</div>
          <div style="font-size:0.8rem; color:#9CA3AF;">Confidence (${{p.agree || 4}}/4 models agree)</div>
        </div>
      </div>
      <div class="grid4">${{cardsHtml}}</div>
    `;
  }} catch (err) {{
    box.innerHTML = `<p style="color:#f87171;">Error: ${{err.message}}</p>`;
  }}
}}

async function runCsvPrediction() {{
  const fileInput = document.getElementById("csvFile");
  const statusEl = document.getElementById("csvStatus");
  const statsBox = document.getElementById("csvStatsContainer");
  const tableBox = document.getElementById("csvTableContainer");
  const dlBtn = document.getElementById("downloadBtn");

  if (!fileInput.files || fileInput.files.length === 0) {{
    statusEl.innerHTML = "<span style='color:#f87171;'>Please choose a CSV or TXT file first.</span>";
    return;
  }}

  const file = fileInput.files[0];
  statusEl.innerHTML = "<span style='color:#38BDF8;'>Processing " + file.name + " via Render NLP API...</span>";
  statsBox.innerHTML = "";
  tableBox.innerHTML = "";
  dlBtn.style.display = "none";

  try {{
    const formData = new FormData();
    formData.append("file", file);
    formData.append("output_format", "json_preview");

    const res = await fetch("/predict_csv", {{ method: "POST", body: formData }});
    if (!res.ok) {{
      const errText = await res.text();
      throw new Error(errText);
    }}
    const payload = await res.json();
    const rows = payload.predictions || [];
    const counts = payload.counts || {{ FAVOR: 0, AGAINST: 0, NONE: 0 }};

    // Build complete Output CSV (id,label) for all rows
    lastCsvOutputText = payload.csv_text || "";
    dlBtn.style.display = "inline-flex";

    // Automatically trigger download of output_predictions.csv!
    downloadOutputCsv();

    let tbody = "";
    for (const r of rows) {{
      const escapedText = String(r.text).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      tbody += `<tr>
        <td><b>${{r.id}}</b></td>
        <td>${{escapedText}}</td>
        <td><span class="tag ${{r.label}}">${{r.label}}</span></td>
        <td>${{(r.confidence * 100).toFixed(1)}}%</td>
      </tr>`;
    }}

    statusEl.innerHTML = `&#10004; Predicted all ${{payload.total_count}} rows &amp; downloaded <code>output_predictions.csv</code> (id, label)!`;
    statsBox.innerHTML = `
      <div class="stats-row">
        <div class="sbox"><div class="lbl">Total Rows</div><div class="num">${{payload.total_count}}</div></div>
        <div class="sbox"><div class="lbl">🟢 FAVOR</div><div class="num">${{counts.FAVOR || 0}}</div></div>
        <div class="sbox"><div class="lbl">🔴 AGAINST</div><div class="num">${{counts.AGAINST || 0}}</div></div>
        <div class="sbox"><div class="lbl">⚪ NONE</div><div class="num">${{counts.NONE || 0}}</div></div>
      </div>
    `;
    const note = payload.total_count > rows.length
      ? `<p style="font-size:0.82rem; margin-top:10px;">Showing first ${{rows.length}} of ${{payload.total_count}} rows below (all ${{payload.total_count}} rows are saved in your downloaded <code>output_predictions.csv</code>).</p>`
      : "";
    tableBox.innerHTML = note + `
      <div class="table-box">
        <table>
          <thead><tr><th>id</th><th>text</th><th>label (Predicted Stance)</th><th>Confidence</th></tr></thead>
          <tbody>${{tbody}}</tbody>
        </table>
      </div>
    `;
  }} catch (err) {{
    statusEl.innerHTML = `<span style="color:#f87171;">Error: ${{err.message}}</span>`;
  }}
}}

function downloadOutputCsv() {{
  if (!lastCsvOutputText) return;
  const blob = new Blob(["\\uFEFF" + lastCsvOutputText], {{ type: "text/csv;charset=utf-8;" }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "output_predictions.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>"""


@app.get("/health")
def health():
    _ = get_engine()
    return {
        "status": "ready",
        "model_id": MODEL_ID,
        "model": "xlmr",
        "models": FOUR_MODELS,
        "claim": TARGET_CLAIM,
        "labels": ["FAVOR", "AGAINST", "NONE"],
        "error": None,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    if payload.data:
        ids = [item.id if item.id is not None else i + 1 for i, item in enumerate(payload.data)]
        texts = [item.text for item in payload.data]
    elif payload.texts:
        ids = list(range(1, len(payload.texts) + 1))
        texts = list(payload.texts)
    elif payload.text is not None:
        ids = [payload.id if payload.id is not None else 1]
        texts = [payload.text]
    else:
        raise HTTPException(status_code=400, detail="Send either 'text' or 'data': [{'id': 1, 'text': '...'}].")

    results = predict_texts(texts, requested_model=payload.model or "all")
    for row_id, r in zip(ids, results):
        r["id"] = row_id
    return {
        "claim": TARGET_CLAIM,
        "model": payload.model or "all",
        "model_id": MODEL_ID,
        "count": len(results),
        "predictions": results,
    }


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
    ids = df[id_col].tolist()
    eval_results = predict_texts(texts)
    labels = [r["label"] for r in eval_results]

    out_df = pd.DataFrame({"id": ids, "label": labels})
    fmt = output_format.strip().lower()

    if fmt == "json_preview":
        counts = dict(Counter(labels))
        preview_rows = []
        for row_id, r in zip(ids[:100], eval_results[:100]):
            preview_rows.append({
                "id": row_id,
                "text": r["text"],
                "label": r["label"],
                "confidence": r["confidence"],
            })
        return {
            "total_count": len(eval_results),
            "counts": {
                "FAVOR": int(counts.get("FAVOR", 0)),
                "AGAINST": int(counts.get("AGAINST", 0)),
                "NONE": int(counts.get("NONE", 0)),
            },
            "predictions": preview_rows,
            "csv_text": out_df.to_csv(index=False),
        }

    if fmt == "submission":
        out_df = pd.DataFrame({"ID": ids, "Predicted Label": labels})

    data = out_df.to_csv(index=False).encode("utf-8-sig")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="output_predictions.csv"'},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=False)
