"""教学演示：芬兰开放数据门户 avoindata.suomi.fi 上的 HSL 共享单车行程分析。

数据来源：Helsinki Regional Transport (HSL), "Helsingin ja Espoon
kaupunkipyörillä ajetut matkat"（赫尔辛基/埃斯波共享单车 OD 数据），
CC BY 4.0, https://avoindata.suomi.fi/data/en_GB/dataset/helsingin-ja-espoon-kaupunkipyorilla-ajatut-matkat

用法：
    pip install pandas matplotlib
    python scripts/analyze_bike_trips.py            # 自动下载 2024-06 数据并出图
    python scripts/analyze_bike_trips.py 2024-07    # 换月份
"""

import sys
import urllib.request
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

# 数据描述文本中公布的月文件 URL 模板（2021 年起每年一季，5—10 月）
URL_TEMPLATE = "https://dev.hsl.fi/citybikes/od-trips-{year}/{year}-{month:02d}.csv"


def ensure_data(month: str) -> Path:
    """下载指定月份 CSV（如已存在则跳过），返回本地路径。"""
    year, m = month.split("-")
    path = DATA_DIR / f"od-trips-{month}.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        url = URL_TEMPLATE.format(year=year, month=int(m))
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, path)
    return path


def load_trips(path: Path) -> pd.DataFrame:
    """读入原始 CSV，做最小清洗：解析时间、换算单位、过滤异常行程。"""
    df = pd.read_csv(path)
    # 列名里的英文缩写换成教学友好的名字，保留原列含义
    df = df.rename(
        columns={
            "Departure": "dep_time",
            "Return": "ret_time",
            "Departure station name": "dep_station",
            "Return station name": "ret_station",
            "Covered distance (m)": "dist_m",
            "Duration (sec.)": "dur_s",
        }
    )
    # 注意：原始数据时间列格式不统一，个别行只有日期没有时间（如 "2024-06-13"）。
    # 这本身就是真实开放数据的典型脏点，用 ISO8601 宽容解析后统一置 NaT 再剔除。
    df["dep_time"] = pd.to_datetime(df["dep_time"], format="ISO8601", errors="coerce")
    df["ret_time"] = pd.to_datetime(df["ret_time"], format="ISO8601", errors="coerce")
    n_bad_time = int(df["dep_time"].isna().sum() + df["ret_time"].isna().sum())
    print(f"时间无法解析的单元格：{n_bad_time}")
    df = df.dropna(subset=["dep_time", "ret_time"])
    df["dist_km"] = df["dist_m"] / 1000
    df["dur_min"] = df["dur_s"] / 60
    df["hour"] = df["dep_time"].dt.hour
    df["weekday"] = df["dep_time"].dt.day_name()
    # 基本质量过滤：非正距离/时长视为脏数据；30 km 以上和 3 小时以上大概率是忘还车
    before = len(df)
    df = df[(df["dist_m"] > 0) & (df["dur_s"] > 0) & (df["dist_km"] < 30) & (df["dur_min"] < 180)]
    print(f"原始 {before} 行，过滤后 {len(df)} 行（剔除 {before - len(df)} 行异常）")
    return df


def describe(df: pd.DataFrame) -> None:
    """描述统计：总量、典型值、Top 站点与 OD 对。"""
    print("\n=== 总体 ===")
    print(f"行程数：{len(df):,}")
    print(f"距离  中位数 {df.dist_km.median():.2f} km，均值 {df.dist_km.mean():.2f} km")
    print(f"时长  中位数 {df.dur_min.median():.1f} min，均值 {df.dur_min.mean():.1f} min")
    same = (df["dep_station"] == df["ret_station"]).mean()
    print(f"借还同站比例：{same:.1%}")

    top = pd.concat([df["dep_station"], df["ret_station"]]).value_counts().head(10)
    print("\n=== 借还总次数 Top10 站点 ===")
    for name, n in top.items():
        print(f"  {name:<30} {n:>7,}")

    od = df.groupby(["dep_station", "ret_station"]).size().nlargest(10)
    print("\n=== 最繁忙 OD 对 Top10 ===")
    for (a, b), n in od.items():
        print(f"  {a} -> {b} : {n:,}")


def plot_hourly(df: pd.DataFrame, month: str) -> None:
    """工作日 vs 周末的出发时刻分布——讲'双峰通勤模式'。"""
    is_weekend = df["dep_time"].dt.weekday >= 5
    fig, ax = plt.subplots(figsize=(9, 5))
    for mask, label, color in [(~is_weekend, "工作日", "#1F2650"), (is_weekend, "周末", "#E47323")]:
        df[mask].groupby("hour").size().reindex(range(24), fill_value=0).plot(
            ax=ax, label=label, color=color, lw=2
        )
    ax.set(title=f"出发时刻分布（{month}）", xlabel="小时", ylabel="行程数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "trips_by_hour.png", dpi=150)


def plot_duration_hist(df: pd.DataFrame, month: str) -> None:
    """时长直方图（截尾 40 分钟）——讲偏态分布与中位数优于均值。"""
    fig, ax = plt.subplots(figsize=(9, 5))
    df[df["dur_min"] < 40]["dur_min"].plot.hist(bins=40, ax=ax, color="#0072BB")
    ax.axvline(df["dur_min"].median(), color="#E47323", lw=2, label=f"中位数 {df.dur_min.median():.1f} min")
    ax.set(title=f"单次行程时长分布（{month}，截尾 40 min）", xlabel="时长（分钟）", ylabel="行程数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "duration_hist.png", dpi=150)


def plot_top_stations(df: pd.DataFrame, month: str) -> None:
    """借还总次数 Top15 站点——讲类别排序条形图。"""
    top = pd.concat([df["dep_station"], df["ret_station"]]).value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    top.sort_values().plot.barh(ax=ax, color="#00B48D")
    ax.set(title=f"借还总次数 Top15 站点（{month}）", xlabel="行程数")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_stations.png", dpi=150)


def plot_daily(df: pd.DataFrame, month: str) -> None:
    """逐日行程数——讲时间序列与事件（如周末、天气讨论引导）。"""
    daily = df.groupby(df["dep_time"].dt.date).size()
    fig, ax = plt.subplots(figsize=(9, 5))
    daily.plot(ax=ax, color="#1F2650", lw=1.8)
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=4))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    ax.set(title=f"逐日行程数（{month}）", xlabel="日期", ylabel="行程数")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "daily_trips.png", dpi=150)


def main() -> None:
    month = sys.argv[1] if len(sys.argv) > 1 else "2024-06"
    OUT_DIR.mkdir(exist_ok=True)
    df = load_trips(ensure_data(month))
    describe(df)
    plot_hourly(df, month)
    plot_duration_hist(df, month)
    plot_top_stations(df, month)
    plot_daily(df, month)
    print(f"\n图表已输出到 {OUT_DIR}")


if __name__ == "__main__":
    main()
