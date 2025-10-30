import streamlit as st
import re, unicodedata, difflib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("AI二次試験 解答自動採点ツール（β）")

# ==========================
# 固定重み（スライダーなし）
# ==========================
W = {
    "intent": 0.30,      # 題意整合
    "coverage": 0.25,    # 論点カバレッジ
    "consistency": 0.20, # 一貫性
    "specificity": 0.15, # 具体性
    "structure": 0.10    # 構成
}

# ==========================
# テキスト前処理
# ==========================
def normalize_ja(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ==========================
# 類似度系
# ==========================
def _tfidf_cosine_pair(a: str, b: str, analyzer="word", ngram=(1,2), min_df=1) -> float:
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=min_df)
    X = vec.fit_transform([a, b])
    A = X[0].toarray(); B = X[1].toarray()
    num = (A * B).sum()
    den = np.linalg.norm(A) * np.linalg.norm(B)
    return float(num / den) if den else 0.0

def hybrid_similarity(a: str, b: str) -> float:
    a = normalize_ja(a); b = normalize_ja(b)
    s_char = _tfidf_cosine_pair(a, b, analyzer="char_wb", ngram=(3,5))
    s_word = _tfidf_cosine_pair(a, b, analyzer="word", ngram=(1,2))
    return 0.7 * s_char + 0.3 * s_word

def best_or_centroid_similarity(user: str, standards: list[str], mode="best") -> float:
    user = normalize_ja(user)
    if not standards:
        return 0.0
    if mode == "best":
        return max(hybrid_similarity(std, user) for std in standards)
    centroid = "\n".join(normalize_ja(s) for s in standards)
    return hybrid_similarity(centroid, user)

# ==========================
# カバレッジ・スコア
# ==========================
def fuzzy_hit(key: str, text: str) -> bool:
    key = normalize_ja(key); text = normalize_ja(text)
    return key in text or difflib.SequenceMatcher(a=key, b=text).ratio() >= 0.80

def coverage_score(keys: list[str], text: str) -> float:
    if not keys:
        return 0.5
    text = normalize_ja(text)
    hits = sum(1 for k in keys if fuzzy_hit(k, text))
    return hits / len(keys)

# ==========================
# 具体性・構成スコア
# ==========================
def specificity_score(text: str) -> float:
    t = normalize_ja(text)
    cues = ["％","%","件","円","日","週","月","前年比","在庫","KPI","ROI","CV","NPV","回転","リードタイム","条件","効果","目的","数値","指標"]
    hit = sum(t.count(c) for c in cues)
    L = max(len(t), 1)
    raw = min(1.0, hit / (L/80))
    return max(0.2, raw)

def structure_score(text: str) -> float:
    t = normalize_ja(text)
    cues = ["①","②","③","・","■","◆","まず","次に","一方","したがって","結果として","理由は","根拠は","課題は","施策は","効果は"]
    hit = sum(t.count(c) for c in cues)
    L = max(len(t), 1)
    raw = min(1.0, 0.2 + hit / (L/100))
    return max(0.2, raw)

# ==========================
# 総合スコア計算
# ==========================
def headline_scores(standards: list[str] | None, user_answer: str | None, keys: list[str] | None):
    standards = standards or []
    user_answer = user_answer or ""
    keys = keys or []
    s_intent = best_or_centroid_similarity(user_answer, standards, mode="best")
    s_cov = coverage_score(keys, user_answer)
    s_cons = s_intent
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

# ==========================
# Streamlit UI
# ==========================
st.subheader("入力欄")

standards_raw = st.text_area("標準解（改行で複数）", height=200)
standards = [s.strip() for s in (standards_raw or "").splitlines() if s.strip()]

keys_raw = st.text_area("論点キーワード（改行で複数）", height=120)
keys = [k.strip() for k in (keys_raw or "").splitlines() if k.strip()]

user_answer = st.text_area("あなたの解答", height=220)

if st.button("評価する", type="primary"):
    scores = headline_scores(standards, user_answer, keys)
    total = total_score(scores)
    s1, s2, s3, s4, s5 = scores

    st.markdown("## 結果")
    st.write(f"**総合スコア：{total:.2f}**")
    st.write(f"- 題意整合　：{s1:.2f}")
    st.write(f"- 論点カバレッジ：{s2:.2f}")
    st.write(f"- 一貫性　　：{s3:.2f}")
    st.write(f"- 具体性　　：{s4:.2f}")
    st.write(f"- 構成　　　：{s5:.2f}")
