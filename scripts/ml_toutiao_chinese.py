"""教学演示：中文新闻标题多标签分类（今日头条数据集）+ Dash 交互界面。

数据：今日头条中文新闻标题分类数据集，382,688 条、15 类，GitHub 镜像：
https://github.com/BenDerPan/toutiao-text-classfication-dataset
原始仓库已删除，本脚本内置下载地址。

演示点：
  - 中文分词（jieba）+ TF-IDF
  - 伪多标签构造（真实标签 + 语义相近类别，模拟"一篇新闻跨多个栏目"）
  - OneVsRest 多标签分类、precision@k 评估
  - Dash 交互界面：输入任意标题 → 实时预测多标签概率条

用法：
    pip install jieba dash scikit-learn
    python scripts/ml_toutiao_chinese.py            # 训练 + 启动看板 :8052
    python scripts/ml_toutiao_chinese.py --train-only   # 只训练出指标
"""

import argparse
import functools
import urllib.request
import zipfile
from pathlib import Path

import jieba
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
URL = "https://raw.githubusercontent.com/BenDerPan/toutiao-text-classfication-dataset/master/toutiao_cat_data.txt.zip"
N_TRAIN = 30000   # 训练子集大小（全部 38 万太大）
VOCAB = 40000     # TF-IDF 词典上限
EXTRA_LABELS = 2  # 每条样本额外挂的伪标签数


def ensure_data() -> Path:
    path = DATA_DIR / "toutiao_cat_data.txt"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        zip_path = DATA_DIR / "toutiao_cat_data.zip"
        print("下载", URL)
        urllib.request.urlretrieve(URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
    return path


def load_df() -> pd.DataFrame:
    path = ensure_data()
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        parts = line.strip().split("_!_")
        if len(parts) >= 4 and parts[3].strip():
            rows.append({"id": parts[0], "cat": parts[2], "title": parts[3]})
    df = pd.DataFrame(rows)
    print(f"加载 {len(df):,} 条，{df['cat'].nunique()} 类")
    return df


def build_multilabel(df: pd.DataFrame) -> pd.DataFrame:
    """用 TF-IDF + 类别质心相似度给每条样本补伪标签。

    对每条标题，在除真实类外再挂 EXTRA_LABELS 个最相近的类别，
    模拟"一篇财经新闻也常出现在科技/文化栏目"的真实场景。
    """
    print("分词 + TF-IDF 向量化…")
    corpus = df["title"].tolist()
    vec = TfidfVectorizer(
        tokenizer=jieba.lcut, max_features=VOCAB, token_pattern=None
    )
    X = vec.fit_transform(corpus)

    # 每类质心：该类别所有样本 TF-IDF 的平均向量（保持稀疏）
    cats = sorted(df["cat"].unique())
    cat_idx = {c: i for i, c in enumerate(cats)}
    centroids = np.zeros((len(cats), X.shape[1]))
    for c in cats:
        mask = (df["cat"] == c).values
        centroids[cat_idx[c]] = X[mask].mean(axis=0).A1

    # 稀疏行归一化，不 dense 化：用 scipy sparse 直接算余弦相似度
    print("计算类别质心相似度，构造伪多标签…")
    import scipy.sparse as sp
    row_norms = np.asarray(X.multiply(X).sum(axis=1)).ravel() ** 0.5 + 1e-9
    Xn = X.multiply(sp.csr_matrix(1.0 / row_norms[:, None])).tocsr()
    cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ cn.T   # sparse × dense → dense，n_samples × n_classes

    labels = []
    for i, c in enumerate(df["cat"]):
        own = cat_idx[c]
        order = np.argsort(-sim[i])
        extra = [cats[j] for j in order if j != own][:EXTRA_LABELS]
        labels.append(sorted(set([c] + extra)))
    df["labels"] = labels
    df["text_vec"] = list(X)   # 存稀疏行，后面直接切片
    return df, vec


def train(df: pd.DataFrame):
    X = np.vstack([v.toarray() for v in df["text_vec"]]) if False else df["text_vec"].tolist()
    # 更快：直接重算一次稀疏矩阵行
    vec = TfidfVectorizer(
        tokenizer=jieba.lcut, max_features=VOCAB, token_pattern=None
    )
    X = vec.fit_transform(df["title"].tolist())

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["labels"])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练 {Xtr.shape[0]:,} / 测试 {Xte.shape[0]:,}，特征 {Xtr.shape[1]:,}，标签 {ytr.shape[1]}")

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200, solver="liblinear"), n_jobs=1
    )
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)
    pred = np.argsort(-proba, axis=1)

    def p_at_k(k):
        hit = 0.0
        for p, t in zip(pred[:, :k], yte):
            true_idx = set(np.where(t)[0])
            hit += len(set(p) & true_idx) / min(k, len(true_idx) or 1)
        return hit / len(yte)

    print(f"precision@1 = {p_at_k(1):.3f}   precision@3 = {p_at_k(3):.3f}   precision@5 = {p_at_k(5):.3f}")
    return vec, mlb, clf


CAT_CN = {
    "news_culture": "文化", "news_entertainment": "娱乐", "news_sports": "体育",
    "news_finance": "财经", "news_house": "房产", "news_car": "汽车",
    "news_edu": "教育", "news_tech": "科技", "news_military": "军事",
    "news_travel": "旅游", "news_world": "国际", "news_agriculture": "农业",
    "news_game": "游戏", "stock": "股市", "news_story": "故事",
}


def serve(vec, mlb, clf) -> None:
    from dash import Dash, Input, Output, dcc, html
    import plotly.graph_objects as go

    @functools.lru_cache(maxsize=200)
    def predict(text: str):
        X = vec.transform([text])
        proba = clf.predict_proba(X)[0]
        order = np.argsort(-proba)
        labels = [(mlb.classes_[i], float(proba[i])) for i in order[:8]]
        return labels

    app = Dash(__name__, title="中文新闻标题多标签分类演示")
    app.layout = html.Div(
        style={"fontFamily": "Microsoft YaHei, sans-serif", "maxWidth": "880px",
               "margin": "0 auto", "padding": "24px"},
        children=[
            html.H2("中文新闻标题 · 多标签分类演示", style={"color": "#1F2650"}),
            html.P([
                "数据：今日头条中文新闻标题分类数据集（15 类，30k 子集训练），",
                html.Br(),
                "真实标签 + TF-IDF 类别质心相似度补伪多标签，模拟「一稿多栏目」。",
                html.Br(),
                "模型：jieba 分词 → TF-IDF → OneVsRest + LogisticRegression。",
            ], style={"color": "#555"}),
            dcc.Input(
                id="q", type="text", placeholder="输入一条新闻标题，回车预测…",
                style={"width": "100%", "padding": "12px", "fontSize": "16px"},
                debounce=True, value="神舟十九号载人飞船发射取得圆满成功",
            ),
            html.Div(id="out", style={"marginTop": "24px"}),
            html.Hr(),
            html.P("数据源：Toutiao text classification dataset (GitHub 镜像)。教学演示，伪多标签由类别质心相似度生成。", style={"color": "#999", "fontSize": "12px"}),
        ],
    )

    @app.callback(Output("out", "children"), Input("q", "value"))
    def update(text):
        if not text or not text.strip():
            return "请输入标题"
        labels = predict(text.strip())
        fig = go.Figure(go.Bar(
            x=[p for _, p in labels][::-1],
            y=[CAT_CN.get(l, l) for l, _ in labels][::-1],
            orientation="h",
            marker_color="#E47323" if labels else "#999",
        ))
        fig.update_layout(
            title="预测标签概率 Top8",
            xaxis_title="概率", yaxis_title="", height=340,
            margin=dict(l=80, r=30, t=50, b=40),
        )
        return dcc.Graph(figure=fig)

    app.run(port=8052, debug=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()

    df = load_df()
    df = df.sample(n=N_TRAIN, random_state=42).reset_index(drop=True)
    df, vec = build_multilabel(df)
    vec2, mlb, clf = train(df)

    if not args.train_only:
        serve(vec2, mlb, clf)


if __name__ == "__main__":
    main()
