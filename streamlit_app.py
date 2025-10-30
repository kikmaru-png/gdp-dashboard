# streamlit_app.py
# ＝＝＝ 模範解答 “類似度だけ” で採点する軽量アプリ ＝＝＝
# - TAC/LEC/大原/KEC/AAS などの模範解答（最大5件）を入力
# - 受験者解答との類似度を複合指標でスコア化（0〜100）
# - 指標：重心コサイン / 上位2平均 / キーワードJaccard / ROUGE-L
# - サイドバーで重み・キーワード数を調整

import math
import re
from collections import Counter
from typing import List, Dict, Tuple
import streamlit as st

# -------------------------
# 前処理・ベクトル化まわり
# -------------------------
def _clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)  # 記号除去
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _tokenize(t: str) -> List[str]:
    return _clean(t).split()

def _build_tfidf_matrix(docs: List[str]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    tokenized = [ _tokenize(d) for d in docs ]
    df = Counter()
    for toks in tokenized:
        for w in set(toks):
            df[w] += 1
    N = len(docs)
    idf = { w: math.log( (N + 1) / (df[w] + 1) ) + 1.0 for w in df }  # smooth

    tfidf_docs = []
    for toks in tokenized:
        tf = Counter(toks)
        vec = { w: (tf[w] / max(1, len(toks))) * idf.get(w, 0.0) for w in tf }
        # L2 正規化
        norm = math.sqrt(sum(v*v for v in vec.values())) or 1.0
        vec = { w: v/norm for w, v in vec.items() }
        tfidf_docs.append(vec)
    return tfidf_docs, idf

def _cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    # 交差キーで十分（高速化）
    keys = a.keys() if len(a) < len(b) else b.keys()
    s = sum(a.get(k,0.0)*b.get(k,0.0) for k in keys)
    # 数値ゆらぎ防止
    return max(0.0, min(1.0, s))

def _top_k_keywords(vec: Dict[str,float], k:int=10) -> set:
    return set([w for w,_ in sorted(vec.items(), key=lambda x: x[1], reverse=True)[:k]])

def _jaccard(a:set, b:set) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b) or 1
    return inter/union

def _lcs(a_tokens: List[str], b_tokens: List[str]) -> int:
    # メモリ控えめ2行DP
    if not a_tokens or not b_tokens: return 0
    prev = [0]*(len(b_tokens)+1)
    for i in range(1, len(a_tokens)+1):
        cur = [0]*(len(b_tokens)+1)
        ai = a_tokens[i-1]
        for j in range(1, len(b_tokens)+1):
            if ai == b_tokens[j-1]:
                cur[j] = prev[j-1] + 1
            else:
                cur[j] = max(prev[j], cur[j-1])
        prev = cur
    return prev[-1]

def _rouge_l(a: str, b: str) -> float:
    a_tok, b_tok = _tokenize(a), _tokenize(b)
    l = _lcs(a_tok, b_tok)
    prec = l / max(1, len(a_tok))
    rec  = l / max(1, len(b_tok))
    if prec==0 and rec==0: return 0.0
    return (2*prec*rec)/(prec+rec)

# -------------------------
# 類似度合成スコア
# -------------------------
def similarity_panel_score(standards: List[str], user_answer: str,
                           w_centroid=0.40, w_top2=0.30, w_jacc=0.15, w_rouge=0.15,
                           k_keywords=12) -> Dict[str, float]:
    """
    standards: 模範解答のリスト（空文字は自動除外）
    user_answer: 評価する受験者解答
    返り値: total/内訳/各社との個別スコア
    """
    # 空の標準は除外
    standards = [s for s in standards if isinstance(s, str) and s.strip()]
    if not standards or not user_answer.strip():
        return {
            "total": 0.0,
            "centroid_sim": 0.0,
            "top2_mean": 0.0,
            "jaccard": 0.0,
            "rougeL": 0.0,
            "per_standard": []
        }

    # ベクトル化
    docs = standards + [user_answer]
    tfidf_vecs, _ = _build_tfidf_matrix(docs)
    std_vecs = tfidf_vecs[:-1]
    user_vec = tfidf_vecs[-1]

    # 1) 重心コサイン
    centroid = {}
    for v in std_vecs:
        for k,vv in v.items():
            centroid[k] = centroid.get(k,0.0) + vv
    centroid = {k: vv/len(std_vecs) for k,vv in centroid.items()}
    norm = math.sqrt(sum(vv*vv for vv in centroid.values())) or 1.0
    centroid = {k: vv/norm for k,vv in centroid.items()}
    centroid_sim = _cosine(centroid, user_vec)

    # 2) 上位2平均
    sims = [_cosine(v, user_vec) for v in std_vecs]
    sims_sorted = sorted(sims, reverse=True)
    top2_mean = sum(sims_sorted[:2]) / max(1, len(sims_sorted[:2]))

    # 3) キーワードJaccard（平均）
    user_kw = _top_k_keywords(user_vec, k_keywords)
    std_kw = [_top_k_keywords(v, k_keywords) for v in std_vecs]
    jaccs = [_jaccard(user_kw, kw) for kw in std_kw]
    jaccard_mean = sum(jaccs)/max(1,len(jaccs))

    # 4) ROUGE-L（平均）
    rouge_scores = [_rouge_l(user_answer, s) for s in standards]
    rouge_mean = sum(rouge_scores)/max(1,len(rouge_scores))

    # 総合（0–1）
    w_sum = max(1e-9, (w_centroid + w_top2 + w_jacc + w_rouge))
    total = (w_centroid*centroid_sim + w_top2*top2_mean + w_jacc*jaccard_mean + w_rouge*rouge_mean) / w_sum

    # 各社との個別
    per = []
    for idx, s in enumerate(standards):
        per.append({
            "index": idx+1,
            "cosine": round(sims[idx], 4),
            "rougeL": round(rouge_scores[idx], 4),
            "jaccard_kw": round(jaccs[idx], 4)
        })

    return {
        "total": round(total, 4),
        "centroid_sim": round(centroid_sim, 4),
        "top2_mean": round(top2_mean, 4),
        "jaccard": round(jaccard_mean, 4),
        "rougeL": round(rouge_mean, 4),
        "per_standard": per
    }

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="二次・答案 類似度スコア", layout="wide")
st.title("📊 二次試験・答案 類似度スコア（模範解答ベース）")

with st.sidebar:
    st.header("⚙️ 設定")
    w_centroid = st.slider("重心コサインの重み", 0.0, 1.0, 0.40, 0.01)
    w_top2     = st.slider("上位2平均の重み",   0.0, 1.0, 0.30, 0.01)
    w_jacc     = st.slider("キーワードJaccardの重み", 0.0, 1.0, 0.15, 0.01)
    w_rouge    = st.slider("ROUGE-Lの重み", 0.0, 1.0, 0.15, 0.01)
    k_keywords = st.slider("キーワード数（Jaccard用）", 5, 30, 12, 1)
    st.caption("※ 合計は内部で自動正規化します（合計≠1でもOK）")

st.markdown("**使い方**：模範解答（最大5件）と、評価したい答案を入力し「採点する」。TAC答案が80点近く出るように、重みやキーワード数を微調整してください。")

cols = st.columns(5)
with cols[0]:
    tac = st.text_area("TAC", height=180, placeholder="木育を実践する場である保育・教育施設における実証実験による新たなアイデア獲得の機会創出、大手ECサイトへの出店による販売チャネルの拡大、社長の子息の経営学の知識やX事業での経験による、SNSを活用した情報発信や子育てイベントへの出展などの積極的な企画・実行など、ターゲット層を意識した内容を実施した。")
with cols[1]:
    lec = st.text_area("LEC", height=180, placeholder="取組は、直営店やアンテナショップ、イベントや保育・教育施設、大手ECサイトやSNS、ワークショップである。工夫は、公共団体との良好な関係や社長の子息のX事業での経験を活かし、PRや情報発信を強化した。また、顧客が製品に触れる機会や学生との共同研究機械を作り、市場の成長可能性や新たなアイデアを探索した。")
with cols[2]:
    ohara = st.text_area("大原", height=180, placeholder="行った取り組みは、従来のルートとは異なる新たな販売チャネルの構築である。20代から40代の教育熱心な子育て家庭との接点をもつために、大手ECサイトへ出店した。工夫は、①SNSを活用した情報発信や子育てイベントへの出展等による認知度向上、②地元の大学との教育連携の推進による製品開発サイクルのj加速う、等である。")
with cols[3]:
    kec = st.text_area("KEC", height=180, placeholder="取組みは、自社併設の直営店や県のアンテナショップ・大手ECサイトでの販売の他、SNSを活用した情報発信や子育てイベントへの出展である。工夫は、ターゲットを教育熱心な家庭とし、地元の県や大学との関係を生かした保育・教育施設において子供たちが日常的にA社製品に触れる機会を作り、新製品のアイデアに繋げた。")
with cols[4]:
    aas = st.text_area("AAS/自作（任意）", height=180, placeholder="AASや自作の標準解答を貼り付け")

user_answer = st.text_area("📝 評価したい答案（必須）", height=200, placeholder="あなたの答案をここに貼り付け")

do_score = st.button("🎯 採点する", type="primary")

if do_score:
    standards = [tac, lec, ohara, kec, aas]
    scores = similarity_panel_score(
        standards, user_answer,
        w_centroid=w_centroid, w_top2=w_top2, w_jacc=w_jacc, w_rouge=w_rouge,
        k_keywords=k_keywords
    )

    if not user_answer.strip():
        st.warning("評価対象の答案を入力してください。")
    elif not any(s.strip() for s in standards):
        st.warning("模範解答を1つ以上入力してください。")
    else:
        st.subheader("結果")
        total_pct = int(round(scores["total"]*100))
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            st.metric("総合スコア", f"{total_pct} / 100")
        with c2:
            st.write("**内訳（0〜1）**")
            st.write({
                "重心コサイン": scores["centroid_sim"],
                "上位2平均": scores["top2_mean"],
                "キーワードJaccard": scores["jaccard"],
                "ROUGE-L": scores["rougeL"],
            })
        with c3:
            st.write("**各模範解答との個別スコア（0〜1）**")
            if scores["per_standard"]:
                st.table(scores["per_standard"])
            else:
                st.info("模範解答が未入力です。")

        st.divider()
        st.caption("ヒント：TAC答案をユーザー答案に入れて実行 ⇒ 内訳を見ながら重み・キーワード数を微調整すると、基準合わせが素早くできます。")
