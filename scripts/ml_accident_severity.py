"""教学演示：赫尔辛基交通事故严重度分类（avoindata.suomi.fi / HSY）。

数据集：liikenneonnettomuudet_Helsingissa.csv，2000 年以来赫尔辛基交通事故
记录，字段含事故类型（LAJI）、坐标（pohj_etrs/ita_etrs）、严重度（VAKAV_A）、
年份（VV）。CC BY 4.0, City of Helsinki。
数据集页：https://avoindata.suomi.fi/data/fi/dataset/liikenneonnettomuudet-helsingissa

演示点：类别不均衡、one-hot + 树模型、精度-召回权衡、特征重要性。
用法：python scripts/ml_accident_severity.py
"""

import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
URL = "https://www.hel.fi/static/avoindata/kymp/liikenneonnettomuudet_Helsingissa.csv"


def ensure(path: Path, url: str) -> None:
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        print("下载", url)
        urllib.request.urlretrieve(url, path)


def main() -> None:
    path = DATA_DIR / "liikenneonnettomuudet_Helsingissa.csv"
    ensure(path, URL)

    # 原始文件分号分隔、Latin-1 容错读；少量坐标缺失的行直接剔除
    df = pd.read_csv(path, sep=";", encoding="latin-1")
    df = df.rename(columns={
        "LAJI": "type", "pohj_etrs": "x", "ita_etrs": "y",
        "VAKAV_A": "severity", "VV": "year",
    }).dropna(subset=["x", "y", "severity"])
    print(f"原始 {len(df):,} 条；严重度分布：{df['severity'].value_counts().to_dict()}")

    # 特征：事故类型 one-hot + 年份（趋势）+ 坐标（空间模式）
    X = pd.get_dummies(df[["type", "year", "x", "y"]], columns=["type"], drop_first=True)
    y = df["severity"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"训练 {Xtr.shape[0]:,} / 测试 {Xte.shape[0]:,}，特征 {Xtr.shape[1]}")

    # 基线：直方图梯度提升（对 5 万行稀疏特征高效，不输 RF）
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    print("\n=== 分类报告（目标：severity 1=严重, 2=轻微）===")
    print(classification_report(yte, pred, digits=3))

    # 混淆矩阵图
    cm = confusion_matrix(yte, pred, labels=[1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks([0, 1], ["1=严重", "2=轻微"])
    ax.set_yticks([0, 1], ["1=严重", "2=轻微"])
    ax.set(xlabel="预测", ylabel="真实",
           title="赫尔辛基交通事故严重度混淆矩阵")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accident_confusion.png", dpi=150)

    # 特征重要性（用排列重要性太贵，退化为皮尔逊相关 + 决策树内部信息）
    # 简单替代：单特征逻辑回归系数排序
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    coefs = pd.Series(lr.named_steps["logisticregression"].coef_[0],
                      index=Xtr.columns).sort_values(key=abs, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    coefs.sort_values().plot.barh(ax=ax, color="#0072BB")
    ax.set(title="逻辑回归特征系数 Top15（正负号表示方向）")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accident_feature_importance.png", dpi=150)

    print(f"图表已输出到 {OUT_DIR}")


if __name__ == "__main__":
    main()
