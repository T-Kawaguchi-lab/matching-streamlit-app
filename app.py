import json,re
from pathlib import Path
from typing import Any,Dict,List
import numpy as np,pandas as pd,streamlit as st
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL="intfloat/multilingual-e5-base"
ROLE_PATH="meta.role"
TEXT_KEY_PRIORITY=["match_text.canonical_card_text","match_text","e5_text","e5_passage","e5_query","card_text","canonical_card_text"]

st.set_page_config(page_title="AI↔他分野 推薦 / AI↔Domain Matching",layout="wide")
st.title("AI研究者 ↔ 他分野研究者 推薦 / AI↔Domain Researcher Matching")
st.caption("類似度はE5（query:/passage:）+ normalize_embeddings=True / Similarity computed with E5 embeddings")

APP_DIR=Path(__file__).resolve().parent
DATA_DIR=APP_DIR/"data"

def read_jsonl_from_path(path:Path)->List[Dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
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
    if isinstance(v2,str)and v2.strip():return v2.strip()[:160]+"…" if len(v2.strip())>160 else v2.strip()
    v=get_nested(r,"match_text.canonical_card_text")
    if isinstance(v,str)and v.strip():return v.strip()[:160]+"…" if len(v.strip())>160 else v.strip()
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
        academic=(get_nested(r,"project.academic_challenge_overview")or"").strip()
        impact=(get_nested(r,"project.ai_leverage_and_impact")or"").strip()
        sources=(get_nested(r,"data.sources_and_collection")or"").strip()
        data_types=(get_nested(r,"data.data_types_raw")or get_nested(r,"data.date_typees_raw")or"").strip()
        modalities=_as_list(get_nested(r,"data.modalities"))
        complexity=_as_list(get_nested(r,"data.complexity_raw"))or _as_list(get_nested(r,"data.complexity_flags"))
        task=_as_list(get_nested(r,"needs.task_type_hints"))
        need=_as_list(get_nested(r,"needs.need_ai_category_hints")or get_nested(r,"need_ai_category_hints")or[])

        ja=[]
        if research_field:ja.append(f"私の研究分野は{research_field}です。")
        if themes:ja.append(f"研究テーマは{_join(themes,'、')}です。")
        if academic:ja.append(f"学術的課題は{academic}です。")
        if impact:ja.append(f"AI活用は{impact}です。")
        if sources:ja.append(f"データ収集は{sources}です。")
        if data_types:ja.append(f"データ種別は{data_types}です。")
        if modalities:ja.append(f"モダリティは{_join(modalities,'、')}です。")
        if complexity:ja.append(f"複雑性は{_join(complexity,'、')}です。")
        if task:ja.append(f"タスクは{_join(task,'、')}です。")
        if need:ja.append(f"必要AIは{_join(need,'、')}です。")
        if trios_topics:ja.append(f"研究トピックは{_join(trios_topics,'、')}です。")
        if trios_papers:ja.append(f"論文は{_join(trios_papers,'、')}です。")

        en=[]
        if research_field:en.append(f"My research field is {research_field}.")
        if themes:en.append(f"My themes include {_join(themes)}.")
        if academic:en.append(f"Academic challenge: {academic}.")
        if impact:en.append(f"AI leverage: {impact}.")
        if sources:en.append(f"Data sources: {sources}.")
        if data_types:en.append(f"Data types: {data_types}.")
        if modalities:en.append(f"Modalities: {_join(modalities)}.")
        if complexity:en.append(f"Complexity: {_join(complexity)}.")
        if task:en.append(f"Tasks: {_join(task)}.")
        if need:en.append(f"Needed AI: {_join(need)}.")
        if trios_topics:en.append(f"Topics: {_join(trios_topics)}.")
        if trios_papers:en.append(f"Papers: {_join(trios_papers)}.")

        return("\n".join([" ".join(ja)," ".join(en)])).strip()

    ai=_as_list(get_nested(r,"offers.ai_categories_raw"))
    methods=_as_list(get_nested(r,"offers.methods_keywords")or get_nested(r,"offers.methods_keyword"))
    themes=_as_list(get_nested(r,"offers.current_main_research_themes"))

    ja=[]
    if research_field:ja.append(f"私の研究分野は{research_field}です。")
    if ai:ja.append(f"AI領域は{_join(ai,'、')}です。")
    if methods:ja.append(f"手法は{_join(methods,'、')}です。")
    if themes:ja.append(f"研究テーマは{_join(themes,'、')}です。")
    if trios_topics:ja.append(f"研究トピックは{_join(trios_topics,'、')}です。")
    if trios_papers:ja.append(f"論文は{_join(trios_papers,'、')}です。")

    en=[]
    if research_field:en.append(f"My research field is {research_field}.")
    if ai:en.append(f"AI categories: {_join(ai)}.")
    if methods:en.append(f"Methods: {_join(methods)}.")
    if themes:en.append(f"Research themes: {_join(themes)}.")
    if trios_topics:en.append(f"Topics: {_join(trios_topics)}.")
    if trios_papers:en.append(f"Papers: {_join(trios_papers)}.")

    return("\n".join([" ".join(ja)," ".join(en)])).strip()

def build_id(i:int)->str:return f"R{i:04d}"

@st.cache_resource
def load_model(name:str)->SentenceTransformer:return SentenceTransformer(name)

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
    return{
        "sim_ai_to_other":(ai_q@ot_p.T).astype(np.float32),
        "sim_other_to_ai":(ot_q@ai_p.T).astype(np.float32)
    }

with st.sidebar:
    st.header("データ選択 / Data selection")
    st.caption("既存またはアップロード / Default or upload")
    jsonl_files=sorted([p.name for p in DATA_DIR.glob("*.jsonl")])
    default_jsonl=jsonl_files[0]if jsonl_files else None
    jsonl_mode=st.radio("JSONL読込 / JSONL load",["既存 / existing","アップロード / upload"],index=0)
    selected_jsonl_name=None;uploaded_jsonl=None
    if jsonl_mode.startswith("既存"):
        if default_jsonl:selected_jsonl_name=st.selectbox("JSONL",jsonl_files,index=0)
        else:st.error("JSONLなし / no JSONL")
    else:uploaded_jsonl=st.file_uploader("JSONL upload",type=["jsonl"])
    st.caption(f"Model: {DEFAULT_MODEL}")

if jsonl_mode.startswith("アップロード"):
    if uploaded_jsonl is None:st.stop()
    rows=read_jsonl_from_uploaded(uploaded_jsonl)
else:
    if selected_jsonl_name is None:st.stop()
    rows=read_jsonl_from_path(DATA_DIR/selected_jsonl_name)

records=[]
roles_raw=[]
for i,r in enumerate(rows,start=1):
    rid=build_id(i)
    meta=r.get("meta",{})
    role_raw=get_nested(r,"meta.role")or get_nested(r,"role")
    role_n=normalize_role_value(role_raw)
    roles_raw.append(role_raw)
    embed_text=build_embedding_text_selected_fields(r)
    matched_url=(get_nested(r,"trios.matched_url")or"").strip()
    records.append({"id":rid,"role_norm":role_n,"name":meta.get("name")or"","affiliation":meta.get("affiliation")or"","position":meta.get("position")or"","research_field":meta.get("research_field")or"","summary":summarize_one_line(r),"embed_text":embed_text,"matched_url":matched_url})

df=pd.DataFrame(records)
ai_df=df[df["role_norm"]=="ai_researcher"].reset_index(drop=True)
other_df=df[df["role_norm"]=="other_field_researcher"].reset_index(drop=True)

st.write("### 事前計算 / Precompute")
with st.spinner("計算中（約10分かかります） / computing（It takes about 10 minutes）"):
    mats=precompute_similarity_matrices(DEFAULT_MODEL,ai_df["embed_text"].tolist(),other_df["embed_text"].tolist())
st.success("完了 / done")

st.markdown("### 人物選択 / Select person")

def role_jp(role):return"AI研究者 / AI researcher"if role=="ai_researcher"else"他分野研究者 / Domain researcher"

id_to_label={r["id"]:f'👤 {r["name"]} ｜ {r["affiliation"]} ｜ {r["position"]} ｜ {r["research_field"]} ｜ 【{role_jp(r["role_norm"])}】'for _,r in df.iterrows()}
options=[None]+list(id_to_label.keys())

def format_func(x):return"🔍名前入力 / input name"if x is None else id_to_label[x]

picked_id=st.selectbox("研究者 / researcher",options,format_func=format_func,index=0)
if picked_id is None:st.stop()

picked=df[df["id"]==picked_id].iloc[0]
if picked["role_norm"]=="ai_researcher":
    query_df=ai_df;doc_df=other_df;sim_matrix=mats["sim_ai_to_other"];query_label="AI研究者 / AI";doc_label="他分野研究者 / Domain";sel_idx=int(ai_df.index[ai_df["id"]==picked_id][0])
else:
    query_df=other_df;doc_df=ai_df;sim_matrix=mats["sim_other_to_ai"];query_label="他分野研究者 / Domain";doc_label="AI研究者 / AI";sel_idx=int(other_df.index[other_df["id"]==picked_id][0])

row=query_df.iloc[sel_idx]
st.write("##### 入力データ / Input data")
col1,col2,col3,col4=st.columns(4)

with col1:st.markdown(f"**名前 / Name**<br>{row.get('name','')}",unsafe_allow_html=True)
with col2:st.markdown(f"**区分 / Role**<br>{query_label}",unsafe_allow_html=True)
with col3:
    url=row.get("url","")
    st.markdown(f'**アンケート / Survey**<br><a href="{url}" target="_blank">Open</a>'if url else"**アンケート / Survey**<br>None",unsafe_allow_html=True)
with col4:
    trios=row.get("matched_url","")
    st.markdown(f'**TRIOS**<br><a href="{trios}" target="_blank">Open</a>'if trios else"**TRIOS**<br>None",unsafe_allow_html=True)

embed_text=str(row.get("embed_text",""))
st.write("文字数 / length:",len(embed_text))
st.text_area("embed_text",embed_text,height=250)

sims=sim_matrix[sel_idx]
order_idx=np.argsort(-sims)
res=doc_df.iloc[order_idx].copy()
res.insert(0,"rank",np.arange(1,len(res)+1))
res.insert(1,"similarity",sims[order_idx].astype(float))
show_cols=["rank","similarity","id","name","affiliation","position","research_field","summary","matched_url"]
res_show=res[show_cols].copy()

st.subheader(f"検索結果 / Results ({doc_label}) 件数 / count: {len(res_show)}")
st.dataframe(res_show,use_container_width=True,height=700)

csv_bytes=res_show.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")
st.download_button("CSV download",csv_bytes,"match_results.csv","text/csv")

json_bytes=res_show.to_json(orient="records",force_ascii=False,indent=2).encode("utf-8")
st.download_button("JSON download",json_bytes,"match_results.json","application/json")