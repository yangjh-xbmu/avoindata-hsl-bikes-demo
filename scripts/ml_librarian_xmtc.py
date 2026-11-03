"""教学演示：avoindata.suomi.fi 上的 XMTC（极端多标签文本分类）基线。

数据集：Kysy kirjastonhoitajalta（“问图书馆员”问答语料 TF-IDF 数据矩阵，
芬兰语），CSC – IT Center for Science 发布，ML 专用稀疏矩阵格式
（LIBSVM/SVMlight，28 万特征、312 标签、约 3.6 万样本）。
数据集页：https://avoindata.suomi.fi/data/en_GB/dataset/kysy-kirjastonhoitajalta
许可：CC0

用法：
    pip install scikit-learn scipy
    python scripts/ml_librarian_xmtc.py
"""

import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BASE = "https://avoindata.suomi.fi/data/dataset/ce9fdd58-e128-4755-8155-3709f68cc6d7/resource"
FULL = f"{BASE}/c04fec54-fbd0-4c03-bc0a-03dc09b34881/download/kirjastonhoitaja_all-fi-voikko.sparse"
# 单个 Voikko 全量文件本地切分：前 80% 训练、后 20% 测试，避免 train/test 维度空间不一致
SPLIT = 0.8


def ensure(path: Path, url: str) -> None:
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print("下载", url)
        urllib.request.urlretrieve(url, path)


def topk(pred_cols: np.ndarray, classes: np.ndarray, y_true: list, k: int) -> float:
    """precision@k：预测概率最高的 k 个标签中，命中真实标签的平均比例。"""
    pred_labels = classes[pred_cols[:, :k]]
    hit = 0.0
    for pred, true in zip(pred_labels, y_true):
        hit += len(set(pred) & set(true)) / min(k, len(true) or 1)
    return hit / len(y_true)


def load_sparse(path: Path):
    """读 XMTC 稀疏矩阵：第一行是数据头（标签数 特征数 样本数），需跳过。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    header = text.splitlines()[0].split()
    n_labels, n_features, n_samples = int(header[0]), int(header[1]), int(header[2])
    body_path = path.with_suffix(".body.libsvm")
    if not body_path.exists():
        body_path.write_text("\n".join(text.splitlines()[1:]), encoding="utf-8")
    X, y = load_svmlight_file(str(body_path), n_features=n_features + 1, multilabel=True)
    return X, list(y), n_labels, n_samples


def split(X, y, frac: float):
    n = int(X.shape[0] * frac)
    return X[:n], y[:n], X[n:], y[n:]


def main() -> None:
    full_path = DATA_DIR / "kirjastonhoitaja_all-fi-voikko.sparse"
    ensure(full_path, FULL)

    print("加载全量语料（LIBSVM 稀疏格式，跳过数据头）…")
    X, y, n_labels, n_samples = load_sparse(full_path)
    print(f"  样本 {X.shape[0]:,}，特征 {X.shape[1]:,}，标签空间 {n_labels}，"
          f"平均标签数 {np.mean([len(t) for t in y]):.2f}")

    Xtr, ytr, Xte, yte = split(X, y, SPLIT)
    print(f"  本地切分：训练 {Xtr.shape[0]:,} / 测试 {Xte.shape[0]:,}")

    # 标签空间太大（3150 标签 × 28 万特征 ≈ 9 亿参数，OvR 直接爆内存）
    # 课堂演示只保留出现频率最高的 N 个标签，训练/测试两端同步过滤
    TOP_LABELS = 100
    from collections import Counter
    freq = Counter(l for labels in y for l in labels)
    keep = set(l for l, _ in freq.most_common(TOP_LABELS))
    print(f"  保留 Top {TOP_LABELS} 高频标签（原 {n_labels}）")

    def relabel(labels):
        return [sorted(set(l) & keep) for l in labels]

    ytr, yte = relabel(ytr), relabel(yte)
    # 丢弃过滤后没有任何标签的样本
    tr_mask = np.array([len(t) > 0 for t in ytr])
    te_mask = np.array([len(t) > 0 for t in yte])
    Xtr, ytr = Xtr[tr_mask], [t for t, m in zip(ytr, tr_mask) if m]
    Xte, yte = Xte[te_mask], [t for t, m in zip(yte, te_mask) if m]
    print(f"  过滤后：训练 {Xtr.shape[0]:,} / 测试 {Xte.shape[0]:,}")

    # 标签二值化：sklearn 新版要求 binary indicator，不再支持 list of lists
    mlb = MultiLabelBinarizer()
    ytr_bin = mlb.fit_transform(ytr)
    yte_bin = mlb.transform(yte)

    # 基线：OneVsRest + LogisticRegression（liblinear，适合高维稀疏）
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=200, solver="liblinear"), n_jobs=4
    )
    print("训练 OneVsRest + LogisticRegression（liblinear）…")
    clf.fit(Xtr, ytr_bin)

    # 输出每个样本概率最高的标签序列
    proba = clf.predict_proba(Xte)
    pred = np.argsort(-proba, axis=1)
    p1 = topk(pred, mlb.classes_, yte, 1)
    p5 = topk(pred, mlb.classes_, yte, 5)
    print(f"\nprecision@1 = {p1:.3f}   precision@5 = {p5:.3f}")
    print("基线到此为止；下一步可试 TF-IDF 特征 + LinearSVC / hashing trick / XGBoost")


if __name__ == "__main__":
    main()
