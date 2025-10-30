# app.py
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="二次対策：答案セルフ採点MVP", layout="wide")

st.title("二次試験・答案セルフ採点（MVP）")
st.caption("学習支援用。実得点・合否は保証しません。")

with st.sidebar:
    st.header("問題メタ")
    year = st.selectbox("年度", ["R7(仮)", "R6", "R5", "R4", "カスタム"])
    jirei = st.selectbox("事例", ["Ⅰ 組織人事", "Ⅱ マーケ", "Ⅲ 生産", "Ⅳ 財務", "カスタム"])
    q = st.text_input("設問（任意）", value="設問X：〜〜について、…を述べよ")
    st.markdown("---")
    st.subheader("採点ルーブリック（重み）")
    w1 = st.slider("題意整合", 0.0, 1.0, 0.25)
    w2 = st.slider("論点カバレッジ", 0.0, 1.0, 0.25)
    w3 = st.slider("根拠一貫性", 0.0, 1.0, 0.20)
    w4 = st.slider("具体性", 0.0, 1.0, 0.15)
    w5 = st.slider("日本語/構成", 0.0, 1.0, 0.15)

st.subheader("標準解（統合・再構築版）")
std_answer = st.text_area(
    "※各校の模範解答の転載は避け、あなたが要約した“標準解”を貼る",
    height=180,
    placeholder="例）与件の強みA/制約Bに基づき、施策①…②…（効果：…、条件：…）"
)

st.subheader("あなたの答案")
user_answer = st.text_area(
    "答案を貼り付け",
    height=220,
    placeholder="（ここに自分の解答を貼る）"
)

# 重要語句の抽出（簡易）：標準解から名詞っぽいキーフレーズを手動入力できる欄も用意
st.markdown("### 重要語句（任意・カバレッジ判定に使用）")
key_phrases_raw = st.text_input(
    "カンマ区切りで入力（例：在庫管理, 部門間連携, 標準化, 指導体制）", value=""
)
key_phrases = [k.strip() for k in key_phrases_raw.split(",") if k.strip()]

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

def headline_scores(std, usr, keys):
    s1 = tfidf_similarity(std, usr)           # 題意整合（簡易代替）
    s2 = coverage_score(keys, usr)            # 論点カバレッジ
    s3 = reasoning_consistency(std, usr)      # 根拠一貫性（簡易）
    s4 = specificity_score(usr)               # 具体性
    s5 = structure_score(usr)                 # 日本語/構成
    return s1, s2, s3, s4, s5

col1, col2 = st.columns(2)
if st.button("採点する", type="primary", use_container_width=True):
    s1, s2, s3, s4, s5 = headline_scores(std_answer, user_answer, key_phrases)
    total = (w1*s1 + w2*s2 + w3*s3 + w4*s4 + w5*s5) / max(w1+w2+w3+w4+w5, 1e-9)
    with col1:
        st.metric("総合スコア（0〜1）", f"{total:.2f}")
        st.progress(total)
    with col2:
        st.write("**内訳**")
        st.write(f"- 題意整合（TF-IDF類似）: {s1:.2f}")
        st.write(f"- 論点カバレッジ（重要語句ヒット率）: {s2:.2f}")
        st.write(f"- 根拠一貫性（簡易）: {s3:.2f}")
        st.write(f"- 具体性（数値/条件の手がかり）: {s4:.2f}")
        st.write(f"- 日本語/構成（箇条書き・接続詞）: {s5:.2f}")

    # 重要語句のヒット/未ヒットを可視化
    if key_phrases:
        st.markdown("### 重要語句のヒット状況")
        hits = [k for k in key_phrases if k in user_answer]
        misses = [k for k in key_phrases if k not in user_answer]
        st.success("🟢 ヒット: " + (", ".join(hits) if hits else "なし"))
        st.error("🔴 未ヒット: " + (", ".join(misses) if misses else "なし"))

st.markdown("---")
st.caption("© 学習支援MVP / 二次の実得点は公表採点のみが正です。標準解は著作権配慮のため再構築版を使用してください。")
