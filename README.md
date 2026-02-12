# JSONL（meta.role）対応 マッチングアプリ

このJSONLは 1行が次のような構造：
- meta.role に 'AI_researcher' などが入る

本アプリは `roleのパス`（デフォルト meta.role）でAI研究者と他分野研究者を分離します。

## 起動
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 使い方
- サイドバーで JSONL を選択
- roleのパス（通常 meta.role）と role値（例: AI_researcher / Other_field_researcher 等）を設定
- 検索方向とTop-Kを選択して検索
