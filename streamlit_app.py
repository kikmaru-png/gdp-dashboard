# app.py
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import difflib
import unicodedata, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def fuzzy_hit(key: str, text: str) -> bool:
    key = normalize_ja(key); text = normalize_ja(text)
    return key in text or difflib.SequenceMatcher(a=key, b=text).ratio() >= 0.80

def coverage_score(keys: list[str], text: str) -> float:
    if not keys:
        return 0.5  # キー未設定なら中間値
    text = normalize_ja(text)
    hits = sum(1 for k in keys if fuzzy_hit(k, text))
    return hits / len(keys)


st.set_page_config(page_title="二次対策：答案セルフ採点MVP", layout="wide")

st.title("二次試験・答案セルフ採点（MVP）")
st.caption("学習支援用。実得点・合否は保証しません。")

with st.sidebar:
    st.header("問題メタ")
    year = st.selectbox("年度", ["R7(仮)", "R6", "R5", "R4", "カスタム"])
    jirei = st.selectbox("事例", ["Ⅰ 組織人事", "Ⅱ マーケ", "Ⅲ 生産", "Ⅳ 財務", "カスタム"])
    q = st.text_input("設問（任意）", value="設問X：〜〜について、…を述べよ")

st.subheader("標準解（統合・再構築版）")
standards_raw = st.text_area("標準解（TAC/LEC/MMC等、改行で複数貼付）", height=200)
standards = [s for s in (standards_raw or "").splitlines() if s.strip()]

# 固定重み（スライダーは使わない）
W = {
    "intent": 0.30,      # 題意整合（標準解との近さ）
    "coverage": 0.25,    # 論点カバレッジ
    "consistency": 0.20, # 一貫性（当面は intent と同値で良い）
    "specificity": 0.15, # 具体性（数値・指標など）
    "structure": 0.10    # 日本語/構成（箇条書き/接続詞など）
}

def tfidf_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    X = vec.fit_transform([a, b])
    v0, v1 = X[0].toarray(), X[1].toarray()
    num = (v0 * v1).sum()
    den = np.linalg.norm(v0) * np.linalg.norm(v1)
    return float(num/den) if den != 0 else 0.0

def coverage_score(keys, text):
    if not keys:
        return 0.5  # キー未設定なら中立値
    hit = sum(1 for k in keys if k in text)
    return hit / len(keys)

def specificity_score(text):
    # ざっくり：数字・固有名詞・条件語が多いほど高く
    cues = ["％","%","件","円","日","週","月","前年比","在庫","KPI","ROI","CV","NPV","回転","リードタイム","条件","効果","目的","数値","指標"]
    hit = sum(text.count(c) for c in cues)
    L = max(len(text), 1)
    return min(1.0, hit / (L/80))  # テキトー正規化

def structure_score(text):
    # 箇条書き・接続詞で加点
    cues = ["①","②","③","・","■","◆","まず","次に","一方","したがって","結果として","理由は","根拠は"]
    hit = sum(text.count(c) for c in cues)
    L = max(len(text), 1)
    return min(1.0, (0.2 + hit / (L/100)))

def reasoning_consistency(std, usr):
    # TF-IDF類似で代用（与件整合は本来与件テキストも必要）
    return tfidf_similarity(std, usr)

# === 標準解（複数行を配列へ） ===
standards_raw = st.text_area("標準解（改行で複数）", height=200)
standards = [s.strip() for s in (standards_raw or "").splitlines() if s.strip()]

# === 論点キーワード（改行で複数） ===
keys_raw = st.text_area("論点キーワード（改行で複数）", height=120)
keys = [k.strip() for k in (keys_raw or "").splitlines() if k.strip()]

# === 受験者の解答 ===
user_answer = st.text_area("あなたの解答", height=220)

def headline_scores(standards: list[str], user_answer: str, keys: list[str]):
    standards = standards or []
    user_answer = user_answer or ""
    keys = keys or []
    # 以降、類似度・カバレッジなどの計算
    
    # 題意整合（＝標準解との近さ；とりあえず best を採用）
    s_intent = best_or_centroid_similarity(user_answer, standards, mode="best")
    # 論点カバレッジ
    s_cov = coverage_score(keys, user_answer)
    # 一貫性（とりあえず intent と同値。将来は別指標に分離）
    s_cons = s_intent
    # 具体性／構成
    s_spec = specificity_score(user_answer)
    s_struct = structure_score(user_answer)
    return s_intent, s_cov, s_cons, s_spec, s_struct

def total_score(scores_tuple):
    s_intent, s_cov, s_cons, s_spec, s_struct = scores_tuple
    return (W["intent"] * s_intent +
            W["coverage"] * s_cov +
            W["consistency"] * s_cons +
            W["specificity"] * s_spec +
            W["structure"] * s_struct)

col1, col2 = st.columns(2)

user_answer = st.text_area("あなたの解答", height=220)

if st.button("評価する", type="primary"):
    scores = headline_scores(standards, user_answer, keys)
    total = total_score(scores)
    s1,s2,s3,s4,s5 = scores

    st.markdown("### 結果")
    st.write(f"総合スコア: **{total:.2f}**")
    st.write(f"- 題意整合: {s1:.2f}")
    st.write(f"- 論点カバレッジ: {s2:.2f}")
    st.write(f"- 一貫性: {s3:.2f}")
    st.write(f"- 具体性: {s4:.2f}")
    st.write(f"- 構成: {s5:.2f}")

st.markdown("---")
st.caption("© 学習支援MVP / 二次の実得点は公表採点のみが正です。標準解は著作権配慮のため再構築版を使用してください。")


def normalize_ja(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tfidf_cosine_pair(a: str, b: str, analyzer="word", ngram=(1,2), min_df=1) -> float:
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df)
    X = vec.fit_transform([a, b])
    A = X[0].toarray(); B = X[1].toarray()
    num = (A * B).sum()
    den = np.linalg.norm(A) * np.linalg.norm(B)
    return float(num / den) if den else 0.0

def hybrid_similarity(a: str, b: str) -> float:
    a = normalize_ja(a); b = normalize_ja(b)
    s_char = _tfidf_cosine_pair(a, b, analyzer="char_wb", ngram=(3,5))  # 日本語に強い
    s_word = _tfidf_cosine_pair(a, b, analyzer="word", ngram=(1,2))     # 英数字/語彙にも対応
    return 0.7 * s_char + 0.3 * s_word

def best_or_centroid_similarity(user: str, standards: list[str], mode="best") -> float:
    user = normalize_ja(user)
    if not standards:
        return 0.0
    if mode == "best":
        return max(hybrid_similarity(std, user) for std in standards)
    centroid = "\n".join(normalize_ja(s) for s in standards)
    return hybrid_similarity(centroid, user)

def specificity_score(text: str) -> float:
    t = normalize_ja(text)
    cues = ["％","%","件","円","日","週","月","前年比","在庫","KPI","ROI","CV","NPV","回転","リードタイム","条件","効果","目的","数値","指標"]
    hit = sum(t.count(c) for c in cues)
    L = max(len(t), 1)
    raw = min(1.0, hit / (L/80))  # 長さに応じて緩く
    return max(0.2, raw)          # 下限0.2

def structure_score(text: str) -> float:
    t = normalize_ja(text)
    cues = ["①","②","③","・","■","◆","まず","次に","一方","したがって","結果として","理由は","根拠は","課題は","施策は","効果は"]
    hit = sum(t.count(c) for c in cues)
    L = max(len(t), 1)
    raw = min(1.0, 0.2 + hit / (L/100))
    return max(0.2, raw)          # 下限0.2
