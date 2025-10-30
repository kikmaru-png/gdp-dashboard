# streamlit_app.py
# -- 二次試験「標準解との差分チェック」簡易採点ツール（修正版オールインワン） --
# 依存: pip install streamlit scikit-learn fugashi[unidic-lite]  (日本語分かち書きが不要ならfugashiは省略可)
# ※fugashi未導入でも動作するように簡易トークナイザを同梱。可能ならfugashi利用を推奨。

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==============
# Tokenizer (JP)
# ==============
def simple_ja_tokenize(text: str) -> List[str]:
    # 記号を削って全角→半角の簡易正規化
    t = text.lower()
    t = re.sub(r'[（）\(\)「」『』【】［］\[\]〈〉…‥，、。．・：；！？!?,\.]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    # ひらがな/カタカナ/漢字/英数を1トークンにまとめる拙速実装
    tokens = re.findall(r'[a-z0-9]+|[ぁ-んァ-ンー]+|[一-龥]+', t)
    return tokens

def ngrams(tokens: List[str], n: int) -> List[str]:
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def featurize(text: str, use_bigrams: bool = True) -> List[str]:
    toks = simple_ja_tokenize(text)
    feats = toks[:]
    if use_bigrams:
        feats += ngrams(toks, 2)
    return feats

def join_feats(feats: List[str]) -> str:
    return ' '.join(feats)

# ==============
# Utility
# ==============
def cosine_sim(a: str, b: str) -> float:
    # TF-IDFでコサイン類似
    vec = TfidfVectorizer(tokenizer=lambda s: s.split(), lowercase=False)
    X = vec.fit_transform([a, b])
    sim = cosine_similarity(X[0], X[1])[0, 0]
    return float(sim)

def centroid_text(docs_as_feat_strings: List[str]) -> str:
    # セントロイド近似：各語のTF-IDF重み合計で上位を代表語にする簡便法
    if not docs_as_feat_strings:
        return ""
    vec = TfidfVectorizer(tokenizer=lambda s: s.split(), lowercase=False)
    X = vec.fit_transform(docs_as_feat_strings)
    vocab = vec.get_feature_names_out()
    # 列方向合計ベクトル（語ごとの総重み）
    w = X.sum(axis=0).A1
    # 上位語を連結して“代表文書”にする
    top_idx = w.argsort()[::-1][:200]  # 上位200語
    rep_terms = [vocab[i] for i in top_idx]
    return ' '.join(rep_terms)

def extract_keyterms_pool(standards: List[str], top_k: int = 25) -> List[str]:
    # 標準解群から、よく出る語(n-gram含む)をキータームとする
    all_feats = []
    for s in standards:
        all_feats += featurize(s)
    cnt = Counter(all_feats)
    # 数字だけ/短すぎるトークンは除外
    cand = [(t, c) for t, c in cnt.items() if not re.fullmatch(r'\d+', t) and len(t) >= 2]
    cand.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in cand[:top_k]]

def coverage_score(user_feats: List[str], keyterms: List[str]) -> float:
    s = set(user_feats)
    hit = sum(1 for k in keyterms if k in s)
    return hit / max(1, len(keyterms))

def structure_score(text: str) -> float:
    # 構成：段落/接続詞/箇条書きなどのヒントを簡易評価
    paras = [p for p in text.split('\n') if p.strip()]
    para_score = min(1.0, len(paras) / 4)  # 4段落で満点
    # 接続語の出現（起承転結っぽいシグナル）
    cues = ['まず', '次に', 'さらに', '一方', 'そのため', 'したがって', '結論', '総括', 'よって']
    cue_count = sum(text.count(c) for c in cues)
    cue_score = min(1.0, cue_count / 6)  # 6回で満点
    # 箇条書きっぽさ
    bullets = re.findall(r'(^|\n)\s*[\-\・\●\▲\d\)①-⑩]+', text)
    bullet_score = 1.0 if bullets else 0.3
    return 0.4*para_score + 0.4*cue_score + 0.2*bullet_score

def specificity_score(text: str) -> float:
    # 具体性：固有名詞っぽい語/数値/具体ワード出現
    nums = len(re.findall(r'\d+[%万円個件点]', text))
    # 具体ワード辞書（最小限）
    concretizers = ['ec', 'sns', 'ワークショップ', 'アンテナショップ', '直営店',
                    '実証', '検証', 'フィードバック', '在庫', '導線', '什器', '価格帯',
                    'アンケート', 'kpi', 'abテスト', '協業', '共同研究', '自治体']
    conc = sum(text.count(w) for w in concretizers)
    # 文長の分散（短長の混在＝具体説明が混じるとやや分散増）を控えめに評価
    sents = re.split(r'[。.!?]', text)
    lengths = [len(s) for s in sents if s.strip()]
    var = (sum((l - (sum(lengths)/max(1,len(lengths))))**2 for l in lengths)/max(1,len(lengths))) if lengths else 0
    var_norm = min(1.0, var/800)  # 適当スケール
    base = min(1.0, (nums*0.15 + conc*0.08))
    return max(0.0, min(1.0, 0.6*base + 0.4*var_norm))

def consistency_score(text: str) -> float:
    # 一貫性：逆接の氾濫・自己矛盾シグナルを軽減（超簡易）
    adversatives = ['しかし', '一方で', 'とはいえ', 'ただし']
    adv = sum(text.count(a) for a in adversatives)
    # 過剰な逆接は一貫性を下げる（2〜3までは許容）
    penalty = max(0.0, (adv - 3) * 0.15)
    # 代名詞過多/指示語過多も減点（曖昧さ）
    shiji = ['これ', 'それ', 'あれ', 'どれ', 'この', 'その', 'あの']
    shiji_count = sum(text.count(s) for s in shiji)
    penalty += max(0.0, (shiji_count - 12) * 0.02)
    return max(0.0, 1.0 - penalty)

def intent_alignment_score(user_feat_str: str, centroid_feat_str: str) -> float:
    # 題意整合：センタロイドとのコサイン
    return cosine_sim(user_feat_str, centroid_feat_str)

def headline_scores(standards: List[str], user_answer: str, keyterms: List[str]) -> Dict[str, float]:
    # 特徴ベクトル化
    std_feat_strs = [join_feats(featurize(s)) for s in standards]
    user_feat_str = join_feats(featurize(user_answer))
    # セントロイド作成
    centroid_str = centroid_text(std_feat_strs)

    # スコア各種
    s_intent = intent_alignment_score(user_feat_str, centroid_str)                   # 題意整合
    s_cover  = coverage_score(featurize(user_answer), keyterms)                      # 論点カバレッジ
    s_consis = consistency_score(user_answer)                                        # 一貫性
    s_spec   = specificity_score(user_answer)                                        # 具体性
    s_struct = structure_score(user_answer)                                          # 構成

    # 標準化（上限丸め）
    s_intent = float(min(1.0, s_intent))
    scores = {
        "題意整合": s_intent,
        "論点カバレッジ": s_cover,
        "一貫性": s_consis,
        "具体性": s_spec,
        "構成": s_struct,
    }

    # 重み付け（任意で調整）
    w = {
        "題意整合": 0.30,
        "論点カバレッジ": 0.30,
        "一貫性": 0.15,
        "具体性": 0.15,
        "構成": 0.10,
    }
    total = sum(scores[k] * w[k] for k in scores.keys())
    scores["総合スコア"] = total
    return scores

# ==============
# UI
# ==============
st.set_page_config(page_title="二次試験・標準解との差分チェック", layout="wide")
st.title("二次試験・標準解との差分チェック（修正版）")

st.markdown("""
- **狙い**：複数の標準解（TAC/LEC/大原/KEC など）を**まとめて参照**し、\
  センタロイド（代表解）と比較することで、**類似語ズレ**でも極端に落ちにくい採点を実現します。  
- **指標**：題意整合（代表解との方向性一致）、論点カバレッジ（標準解由来キーターム網羅）、一貫性、具体性、構成。
""")

with st.expander("標準解プール（ここに各校の模範解答＋多様解を貼ってください）", expanded=True):
    default_pool = """[TAC]
木育を実践する場である保育・教育施設における実証実験による新たなアイデア獲得の機会創出、大手ECサイトへの出店による販売チャネルの拡大、社長の子息の経営学の知識やX事業での経験による、SNSを活用した情報発信や子育てイベントへの出展などの積極的な企画・実行など、ターゲット層を意識した内容を実施した。

[LEC]
取組は、直営店やアンテナショップ、イベントや保育・教育施設、大手ECサイトやSNS、ワークショップである。工夫は、公共団体との良好な関係や社長の子息のX事業での経験を活かし、PRや情報発信を強化した。また、顧客が製品に触れる機会や学生との共同研究機械を作り、市場の成長可能性や新たなアイデアを探索した。

[assistant_A_concise]
直営店・アンテナショップ・イベント出展で接点を拡大し、保育・教育施設での実証（木育）により使用体験とフィードバック取得を図った。大手EC出店とSNS発信で情報到達と購買導線を整備。自治体・公共団体との連携や、学生との共同研究・ワークショップで認知拡大と新アイデア創出を進めた。これらをターゲット別に組み合わせ、顧客接点→体験→購入の流れを一貫化した点が工夫である。

[assistant_B_bold]
顧客仮説（保育・教育関係者／子育て世帯／ギフト層）ごとに体験価値仮説を立て、①保育・教育施設での実証実験で学齢×用途別の受容性を検証、②イベント・ワークショップで触感/安全性の強みを検証、③大手EC＆SNSで検索語・UGCを計測。得られた示唆を商品写真・訴求文・価格帯・同梱物へ高速に反映し、直営店／アンテナショップでAB比較。自治体・学校・学生との共同研究で新デザイン/利用シーンを継続創出。計測→学習→改善のサイクルを複数チャネル横断で回した点が工夫。

[assistant_C_practical]
上流で木育ストーリーを作り、保育・教育施設での導入キット（サンプル・取扱手引・安全資料）を配布→授業/行事で使用→事後アンケートで改善。中流でイベント/ワークショップ台本（体験→制作→写真共有）を標準化し、SNS投稿を促進。下流で大手ECは「年齢別/用途別/ギフト別」ナビ、直営店/アンテナショップは触感訴求の動線什器を整備。自治体コラボで地域木材×X焼の限定企画、学生とプロトタイプ共創。全体をコンテンツ連動（授業→SNS→EC/店頭）に束ね、回遊を生むよう導線設計した点が工夫。
"""
    pool_text = st.text_area("標準解プール（自由に追加/削除OK）：", value=default_pool, height=350)

def parse_pool(raw: str) -> List[str]:
    # [ラベル] セクション毎に取り出し
    # 例: [TAC] 〜本文〜
    blocks = re.split(r'\n\[[^\]]+\]\n', '\n' + raw.strip())
    # 先頭の空split対策
    texts = [b.strip() for b in blocks if b.strip()]
    return texts

with st.expander("（任意）キーターム抽出の上限数", expanded=False):
    top_k = st.slider("標準解から抽出するキーターム数", min_value=10, max_value=80, value=30, step=5)

st.subheader("あなたの解答")
user_answer = st.text_area("ここに答案を貼り付け：", height=220, placeholder="例：直営店とECを統合し、木育の実証から得た示唆をSNSで…")

col1, col2 = st.columns([1,1])
with col1:
    if st.button("採点する", use_container_width=True):
        try:
            standards = parse_pool(pool_text)
            if len(standards) < 2:
                st.error("標準解プールは最低でも2本以上入れてください。")
            elif not user_answer.strip():
                st.error("あなたの解答を入力してください。")
            else:
                # キーターム抽出（標準解群→上位語）
                keyterms = extract_keyterms_pool(standards, top_k=top_k)
                # スコア計算
                scores = headline_scores(standards, user_answer, keyterms)

                # 表示
                st.success("採点完了")
                order = ["総合スコア", "題意整合", "論点カバレッジ", "一貫性", "具体性", "構成"]
                for k in order:
                    v = scores[k]
                    st.write(f"**{k}**：{v:.2f}")

                with st.expander("参考：抽出されたキーターム（標準解由来）"):
                    st.write(", ".join(keyterms))

                st.caption("注：本ツールは**自動評価の補助**です。最終判断は人間の答案レビューで行ってください。")

        except Exception as e:
            st.exception(e)

with col2:
    st.markdown("### 使い方のヒント")
    st.markdown("""
- **標準解プール**にTAC/LEC/大原/KECなど複数の模範解を貼ると、\
  自動で代表解（セントロイド）を作って**題意整合**を判定します。
- **論点カバレッジ**は、標準解から自動抽出したキータームの**ヒット率**を評価します。
- **具体性**は数値・具体語・プロセス語の出現などから近似評価。
- 類似語辞書を作らなくても、**複数標準解の集合**でズレを緩和する設計です。
""")

st.markdown("---")
st.caption("v0.5 修正版：センタロイド採用、スコアの極端化を緩和、UI簡素化。")
