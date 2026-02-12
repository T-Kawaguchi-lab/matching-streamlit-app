import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Prefer E5 via sentence-transformers; fallback to TF-IDF if it fails.
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except Exception:
    from sklearn.feature_extraction.text import TfidfVectorizer
    ST_AVAILABLE = False

st.set_page_config(page_title="AI↔他分野 マッチング（JSONLネストrole対応）", layout="wide")
st.title("AI研究者 ↔ 他分野研究者 マッチング（JSONL / meta.role 対応）")
st.caption("JSONL（1行1JSON）を同フォルダから読み込み、role（例: meta.role）でAI研究者と他分野研究者を分離して類似検索します。")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "intfloat/multilingual-e5-base"


# ------------------------
# JSON helpers
# ------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_nested(d: Dict[str, Any], path: str) -> Any:
    """Get nested value by dot path, e.g., 'meta.role'."""
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def normalize_role_value(v: Any) -> str:
    """Normalize role value to one of: ai_researcher / other_field_researcher / other."""
    if v is None:
        return ""
    s = str(v).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    # Common variants
    if s in {"ai_researcher", "ai", "provider", "system_researcher", "system", "ai_research"}:
        return "ai_researcher"
    if s in {"other_field_researcher", "other", "needs", "science_researcher", "domain_researcher", "non_ai", "other_field"}:
        return "other_field_researcher"
    return s


def first_nonempty(d: Dict[str, Any], keys: List[str]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Try nested common place: match_text.canonical_card_text
    v = get_nested(d, "match_text.canonical_card_text")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return ""


def ensure_prefix(text: str, prefix: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if re.match(r"^\s*(query:|passage:)\s*", t, flags=re.IGNORECASE):
        t = re.sub(r"^\s*(query:|passage:)\s*", prefix + " ", t, flags=re.IGNORECASE)
        return t.strip()
    return f"{prefix} {t}".strip()


# ------------------------
# Embedding
# ------------------------
@st.cache_resource
def load_model(model_name: str):
    return SentenceTransformer(model_name)


def embed_e5(model_name: str, queries: List[str], docs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    model = load_model(model_name)
    q = model.encode([ensure_prefix(t, "query:") for t in queries], normalize_embeddings=True, show_progress_bar=False)
    d = model.encode([ensure_prefix(t, "passage:") for t in docs], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(q), np.asarray(d)


@st.cache_resource
def fit_tfidf(all_texts: List[str]):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X = vec.fit_transform(all_texts)
    return vec, X


def embed_tfidf(queries: List[str], docs: List[str]):
    vec, X = fit_tfidf(queries + docs)
    q = X[:len(queries)]
    d = X[len(queries):]
    return q, d


# ------------------------
# UI: select local jsonl
# ------------------------
with st.sidebar:
    st.header("入力ファイル（同フォルダ）")
    candidates = sorted([p.name for p in APP_DIR.glob("*.jsonl")])
    if not candidates:
        st.error("app.py と同じフォルダに .jsonl が見つかりません。")
        st.stop()

    file_name = st.selectbox("使用するJSONL", candidates, index=0)
    jsonl_path = APP_DIR / file_name

    st.divider()
    st.header("role設定（ネスト対応）")
    st.caption("このJSONLは role が meta.role に入っている想定です。必要なら変更してください。")
    role_path = st.text_input("roleのパス（ドット区切り）", value="meta.role")

    ai_role_value = st.text_input("AI研究者のrole値", value="AI_researcher")
    other_role_value = st.text_input("他分野研究者のrole値", value="other_field_researcher")

    st.divider()
    st.header("テキスト設定")
    text_key_priority = st.multiselect(
        "本文キー候補（優先順）",
        options=["e5_text", "match_text.canonical_card_text", "match_text.one_line_pitch", "e5_passage", "e5_query", "text", "content", "summary"],
        default=["match_text.canonical_card_text", "e5_text", "e5_passage", "e5_query"],
        help="ネストキーは 'match_text.canonical_card_text' のように指定できます。"
    )

    st.divider()
    st.header("検索")
    direction = st.radio("検索方向", ["AI研究者 → 他分野研究者", "他分野研究者 → AI研究者"], index=0)
    top_k = st.slider("Top-K", 3, 50, 10, 1)

    st.divider()
    st.header("モデル")
    model_name = st.text_input("E5モデル名（SentenceTransformers）", value=DEFAULT_MODEL)
    st.caption("E5が読み込めない場合、TF‑IDFに自動フォールバックします。")


# ------------------------
# Load & prepare
# ------------------------
rows = read_jsonl(jsonl_path)
if not rows:
    st.error("JSONLにレコードがありません。")
    st.stop()

# Build a lightweight df of top-level keys (for debugging display)
df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows])
st.write("### JSONLトップレベルキー（参考）")
st.write(sorted(list(rows[0].keys())))

# Extract role from nested path
roles_raw = [get_nested(r, role_path) for r in rows]
roles_norm = [normalize_role_value(v) for v in roles_raw]

ai_norm = normalize_role_value(ai_role_value)
other_norm = normalize_role_value(other_role_value)

# Determine indices
ai_idx = [i for i, rn in enumerate(roles_norm) if rn == ai_norm]
other_idx = [i for i, rn in enumerate(roles_norm) if rn == other_norm]

colA, colB, colC = st.columns(3)
colA.metric("総レコード数", len(rows))
colB.metric("AI研究者", len(ai_idx))
colC.metric("他分野研究者", len(other_idx))

if not ai_idx or not other_idx:
    st.warning("role分離の結果、片側が0件です。role値/roleパスを確認してください。")
    st.write("role_rawのユニーク（先頭30）:", sorted({str(v) for v in roles_raw if v is not None})[:30])
    st.stop()

# Build texts using priority keys (support nested dot keys)
def get_text_by_priority(r: Dict[str, Any], priorities: List[str]) -> str:
    for key in priorities:
        if "." in key:
            v = get_nested(r, key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        else:
            v = r.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # fallback
    return first_nonempty(r, ["e5_text", "e5_passage", "e5_query"])

ai_rows = [rows[i] for i in ai_idx]
other_rows = [rows[i] for i in other_idx]

ai_texts = [get_text_by_priority(r, text_key_priority) for r in ai_rows]
other_texts = [get_text_by_priority(r, text_key_priority) for r in other_rows]

# Direction
if direction == "AI研究者 → 他分野研究者":
    query_rows, query_texts = ai_rows, ai_texts
    doc_rows, doc_texts = other_rows, other_texts
    query_label = "AI研究者（query）"
    doc_label = "他分野研究者（document）"
else:
    query_rows, query_texts = other_rows, other_texts
    doc_rows, doc_texts = ai_rows, ai_texts
    query_label = "他分野研究者（query）"
    doc_label = "AI研究者（document）"

st.divider()

# Build display names (meta.affiliation + meta.research_field + id)
def make_display(r: Dict[str, Any]) -> str:
    rid = get_nested(r, "meta.respondent_id") or get_nested(r, "meta.submission_id") or r.get("id") or ""
    aff = get_nested(r, "meta.affiliation") or ""
    field = get_nested(r, "meta.research_field") or ""
    pos = get_nested(r, "meta.position") or ""
    s = f"{rid}"
    if aff:
        s += f" | {aff}"
    if pos:
        s += f" | {pos}"
    if field:
        s += f" | {field}"
    return s

options = [make_display(r) for r in query_rows]
selected = st.selectbox(f"{query_label} を選択", options)
selected_idx = options.index(selected)

with st.expander("選択したquery本文（埋め込み対象）", expanded=False):
    st.write(query_texts[selected_idx])

run = st.button(f"検索実行：{query_label} → {doc_label}", type="primary")
if not run:
    st.stop()

# Similarity
sims = None
if ST_AVAILABLE:
    try:
        q_vecs, d_vecs = embed_e5(model_name, query_texts, doc_texts)
        sims = cosine_similarity([q_vecs[selected_idx]], d_vecs)[0]
    except Exception as e:
        st.warning(f"E5でエラーが出たためTF‑IDFにフォールバックします: {e}")

if sims is None:
    q_mat, d_mat = embed_tfidf(query_texts, doc_texts)
    sims = cosine_similarity(q_mat[selected_idx], d_mat)[0]

top_k = min(top_k, len(doc_rows))
top_idx = np.argsort(-sims)[:top_k]

st.subheader("検索結果（Top-K）")

for rank, j in enumerate(top_idx, start=1):
    r = doc_rows[int(j)]
    score = float(sims[int(j)])
    st.markdown(f"### {rank}. 類似度: **{score:.4f}**")
    st.write({"respondent_id": get_nested(r, "meta.respondent_id"),
              "affiliation": get_nested(r, "meta.affiliation"),
              "position": get_nested(r, "meta.position"),
              "research_field": get_nested(r, "meta.research_field"),
              "role": get_nested(r, role_path)})
    st.write(doc_texts[int(j)])
    st.markdown("---")

# Download results CSV
out_rows = []
for j in top_idx:
    r = doc_rows[int(j)]
    out_rows.append({
        "similarity": float(sims[int(j)]),
        "respondent_id": get_nested(r, "meta.respondent_id"),
        "submission_id": get_nested(r, "meta.submission_id"),
        "affiliation": get_nested(r, "meta.affiliation"),
        "position": get_nested(r, "meta.position"),
        "research_field": get_nested(r, "meta.research_field"),
        "role": get_nested(r, role_path),
        "text": doc_texts[int(j)]
    })
out_df = pd.DataFrame(out_rows)
csv_bytes = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("結果をCSVでダウンロード（UTF-8 BOM）", data=csv_bytes, file_name="match_results_utf8sig.csv", mime="text/csv")
