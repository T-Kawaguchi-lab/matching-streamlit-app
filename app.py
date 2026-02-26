import json,re
from pathlib import Path
from typing import Any,Dict,List
import numpy as np,pandas as pd,streamlit as st
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL="intfloat/multilingual-e5-base"
ROLE_PATH="meta.role"
TEXT_KEY_PRIORITY=["match_text.canonical_card_text","match_text","e5_text","e5_passage","e5_query","card_text","canonical_card_text"]

st.set_page_config(page_title="AI↔Other Researcher Recommendation / AI研究者↔他分野研究者推薦",layout="wide")
st.title("AI Researcher ↔ Other Field Researcher Recommendation / AI研究者 ↔ 他分野研究者 推薦")
st.caption("Similarity computed using E5(query:/passage:) + normalize_embeddings=True / 類似度はE5(query:/passage:) + normalize_embeddings=Trueで計算")

APP_DIR=Path(__file__).resolve().parent
DATA_DIR=APP_DIR/"data"

def read_jsonl_from_path(path:Path)->List[Dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8",errors="ignore")as f:
        for line in f:
            line=line.strip()
            if line:rows.append(json.loads(line))
    return rows

def read_jsonl_from_uploaded(uploaded)->List[Dict[str,Any]]:
    content=uploaded.getvalue().decode("utf-8",errors="ignore").splitlines()
    return[json.loads(line)for line in content if line.strip()]

def read_csv_from_path(path:Path)->pd.DataFrame:return pd.read_csv(path)
def read_csv_from_uploaded(uploaded)->pd.DataFrame:return pd.read_csv(uploaded)

def get_nested(d:Dict[str,Any],path:str)->Any:
    cur=d
    for part in path.split("."):
        if isinstance(cur,dict)and part in cur:cur=cur[part]
        else:return None
    return cur

def normalize_role_value(v:Any)->str:
    if v is None:return""
    s=str(v).strip().lower().replace(" ","_").replace("-","_")
    if s in{"ai_researcher","ai","provider","system_researcher","system","ai_research","ai-researcher","ai_researchers","ai研究者","ai研究","ai系","ai分野"}:return"ai_researcher"
    if s in{"other_field_researcher","other","needs","science_researcher","domain_researcher","non_ai","other_field","other-field-researcher","domain","他分野研究者","非ai","非_ai","non-ai"}:return"other_field_researcher"
    return s

def ensure_prefix(text:str,prefix:str)->str:
    t=(text or"").strip()
    if not t:return""
    if re.match(r"^\s*(query:|passage:)\s*",t,flags=re.IGNORECASE):
        t=re.sub(r"^\s*(query:|passage:)\s*",prefix+" ",t,flags=re.IGNORECASE)
        return t.strip()
    return f"{prefix} {t}".strip()

def summarize_one_line(r:Dict[str,Any])->str:
    v=get_nested(r,"match_text.one_line_pitch")
    if isinstance(v,str)and v.strip():return v.strip()
    v2=r.get("match_text")
    if isinstance(v2,str)and v2.strip():return(v2[:160]+"…")if len(v2)>160 else v2
    v=get_nested(r,"match_text.canonical_card_text")
    if isinstance(v,str)and v.strip():return(v[:160]+"…")if len(v)>160 else v
    return""

def _as_list(x):
    if x is None:return[]
    if isinstance(x,list):return[str(v).strip()for v in x if str(v).strip()]
    s=str(x).strip()
    return[s]if s else[]

def _join(xs,sep=", "):return sep.join([str(x).strip()for x in xs if str(x).strip()])

def build_embedding_text_selected_fields(r:Dict[str,Any])->str:
    role_raw=(get_nested(r,"meta.role")or get_nested(r,"role")or"").lower()
    is_domain=("domain"in role_raw)or("other"in role_raw)
    research_field=(get_nested(r,"meta.research_field")or r.get("research_field")or"").strip()
    trios_topics=_as_list(get_nested(r,"trios.research_topics"))
    trios_papers=_as_list(get_nested(r,"trios.papers"))

    if is_domain:
        themes=_as_list(get_nested(r,"project.themes"))
        academic_challenge_overview=(get_nested(r,"project.academic_challenge_overview")or"").strip()
        ai_leverage_and_impact=(get_nested(r,"project.ai_leverage_and_impact")or"").strip()
        sources=(get_nested(r,"data.sources_and_collection")or"").strip()
        data_types_raw=((get_nested(r,"data.data_types_raw")or"").strip()or(get_nested(r,"data.date_typees_raw")or"").strip())
        modalities=_as_list(get_nested(r,"data.modalities"))
        complexity_flags=_as_list(get_nested(r,"data.complexity_flags"))
        complexity_raw=_as_list(get_nested(r,"data.complexity_raw"))
        task_type_hints=_as_list(get_nested(r,"needs.task_type_hints"))
        need_ai_hints=_as_list(get_nested(r,"needs.need_ai_category_hints")or get_nested(r,"needs.needed_ai_category_hints")or[])

        ja,en=[],[]
        if research_field:
            ja.append(f"私の研究分野は{research_field}です。");en.append(f"My research field is {research_field}.")
        if themes:
            ja.append(f"研究テーマは{_join(themes,'、')}です。");en.append(f"My research themes include {_join(themes)}.")
        if academic_challenge_overview:
            ja.append(f"学術的課題の概要は{academic_challenge_overview}です。");en.append(f"An overview of my academic challenge is {academic_challenge_overview}.")
        if ai_leverage_and_impact:
            ja.append(f"AI活用の方針・期待するインパクトは{ai_leverage_and_impact}です。");en.append(f"My AI leverage plan and expected impact: {ai_leverage_and_impact}.")
        if sources:
            ja.append(f"データの出所・収集方法は{sources}です。");en.append(f"My data sources/collection include {sources}.")
        if data_types_raw:
            ja.append(f"扱うデータ種別は{data_types_raw}です。");en.append(f"The data types are {data_types_raw}.")
        if modalities:
            ja.append(f"データのモダリティは{_join(modalities,'、')}です。");en.append(f"Data modalities include {_join(modalities)}.")
        if complexity_raw:
            ja.append(f"データの複雑性は{_join(complexity_raw,'、')}です。");en.append(f"The data complexity is {_join(complexity_raw)}.")
        elif complexity_flags:
            ja.append(f"データの複雑性フラグは{_join(complexity_flags,'、')}です。");en.append(f"Complexity flags include {_join(complexity_flags)}.")
        if task_type_hints:
            ja.append(f"想定タスク種別のヒントは{_join(task_type_hints,'、')}です。");en.append(f"Task type hints include {_join(task_type_hints)}.")
        if need_ai_hints:
            ja.append(f"必要とするAI領域のヒントは{_join(need_ai_hints,'、')}です。");en.append(f"Needed AI categories include {_join(need_ai_hints)}.")
        if trios_topics:
            ja.append(f"研究トピックは{_join(trios_topics,'、')}です。");en.append(f"Research topics include {_join(trios_topics)}.")
        if trios_papers:
            ja.append(f"関連論文は{_join(trios_papers,'、')}です。");en.append(f"Related papers include {_join(trios_papers)}.")
        return(" ".join(ja)+"\n"+" ".join(en)).strip()

    ai_categories_raw=_as_list(get_nested(r,"offers.ai_categories_raw"))
    methods_keyword=_as_list(get_nested(r,"offers.methods_keyword")or get_nested(r,"offers.methods_keywords")or[])
    current_main_research_themes=_as_list(get_nested(r,"offers.current_main_research_themes"))

    ja,en=[],[]
    if research_field:
        ja.append(f"私の研究分野は{research_field}です。");en.append(f"My research field is {research_field}.")
    if ai_categories_raw:
        ja.append(f"提供できるAI領域は{_join(ai_categories_raw,'、')}です。");en.append(f"I can offer AI categories such as {_join(ai_categories_raw)}.")
    if methods_keyword:
        ja.append(f"主な手法キーワードは{_join(methods_keyword,'、')}です。");en.append(f"Methods/keywords include {_join(methods_keyword)}.")
    if current_main_research_themes:
        ja.append(f"現在の主な研究テーマは{_join(current_main_research_themes,'、')}です。");en.append(f"My current main research themes include {_join(current_main_research_themes)}.")
    if trios_topics:
        ja.append(f"研究トピックは{_join(trios_topics,'、')}です。");en.append(f"Research topics include {_join(trios_topics)}.")
    if trios_papers:
        ja.append(f"関連論文は{_join(trios_papers,'、')}です。");en.append(f"Related papers include {_join(trios_papers)}.")
    return(" ".join(ja)+"\n"+" ".join(en)).strip()

def build_id(i:int)->str:return f"R{i:04d}"

@st.cache_resource
def load_model(model_name:str)->SentenceTransformer:return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def encode_texts(model_name:str,texts:List[str],mode:str)->np.ndarray:
    model=load_model(model_name)
    pref="query:"if mode=="query"else"passage:"
    prep=[ensure_prefix(t,pref)for t in texts]
    emb=model.encode(prep,normalize_embeddings=True,show_progress_bar=False)
    return np.asarray(emb,dtype=np.float32)

@st.cache_data(show_spinner=False)
def precompute_similarity_matrices(model_name:str,ai_texts:List[str],other_texts:List[str])->Dict[str,np.ndarray]:
    ai_q=encode_texts(model_name,ai_texts,"query")
    ai_p=encode_texts(model_name,ai_texts,"passage")
    ot_q=encode_texts(model_name,other_texts,"query")
    ot_p=encode_texts(model_name,other_texts,"passage")
    return{"sim_ai_to_other":ai_q@ot_p.T,"sim_other_to_ai":ot_q@ai_p.T}

with st.sidebar:
    st.header("Data selection / データ選択")
    st.caption("Default uses data/ in repo. You can replace here. / デフォルトはdata/を使用")
    jsonl_files=sorted([p.name for p in DATA_DIR.glob("*.jsonl")])
    default_jsonl=jsonl_files[0]if jsonl_files else None
    jsonl_mode=st.radio("JSONL load / JSONL読み込み",["Use existing / 既存","Upload / アップロード"],index=0)
    selected_jsonl_name=None;uploaded_jsonl=None
    if jsonl_mode=="Use existing / 既存":
        if default_jsonl is None:st.error("No JSONL. Upload. / JSONLなし")
        else:selected_jsonl_name=st.selectbox("JSONL file / JSONLファイル",jsonl_files,index=0)
    else:uploaded_jsonl=st.file_uploader("Upload JSONL / JSONLアップロード",type=["jsonl"])
    st.divider()
    csv_files=sorted([p.name for p in DATA_DIR.glob("*.csv")])
    default_csv="url_mapping_mock.csv"if"url_mapping_mock.csv"in csv_files else(csv_files[0]if csv_files else None)
    csv_mode=st.radio("CSV load / CSV読み込み",["Use existing / 既存","Upload / アップロード"],index=0)
    selected_csv_name=None;uploaded_csv=None
    if csv_mode=="Use existing / 既存":
        if default_csv is None:st.warning("No CSV. / CSVなし")
        else:selected_csv_name=st.selectbox("CSV file / CSVファイル",csv_files,index=0)
    else:uploaded_csv=st.file_uploader("Upload CSV / CSVアップロード",type=["csv"])
    st.divider()
    st.caption(f"Model / モデル: {DEFAULT_MODEL}")