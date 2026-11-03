"""教学演示：avoindata.suomi.fi / HSL 赫尔辛基共享单车交互式分析看板。

运行：
    pip install dash pandas plotly
    python app.py           # 然后浏览器打开 http://127.0.0.1:8050
"""

import functools
import urllib.request
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

URL_TEMPLATE = "https://dev.hsl.fi/citybikes/od-trips-{year}/{year}-{month:02d}.csv"
STATIONS_URL = (
    "https://opendata.arcgis.com/datasets/"
    "726277c507ef4914b0aec3cbcfcbfafc_0.csv"
)
MONTHS = ["2024-05", "2024-06", "2024-07", "2024-08", "2024-09", "2024-10"]

BRAND = "#1F2650"
ACCENT = "#E47323"


def ensure_data(month: str) -> Path:
    year, m = month.split("-")
    path = DATA_DIR / f"od-trips-{month}.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(
            URL_TEMPLATE.format(year=year, month=int(m)), path
        )
    return path


def ensure_stations() -> Path:
    path = DATA_DIR / "stations.csv"
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(STATIONS_URL, path)
    return path


@functools.lru_cache(maxsize=4)
def load_trips(month: str) -> pd.DataFrame:
    df = pd.read_csv(ensure_data(month))
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
    # 真实脏点：个别行只有日期没有时间，ISO8601 宽容解析后剔除
    df["dep_time"] = pd.to_datetime(df["dep_time"], format="ISO8601", errors="coerce")
    df["ret_time"] = pd.to_datetime(df["ret_time"], format="ISO8601", errors="coerce")
    df = df.dropna(subset=["dep_time", "ret_time"])
    df["dist_km"] = df["dist_m"] / 1000
    df["dur_min"] = df["dur_s"] / 60
    df["hour"] = df["dep_time"].dt.hour
    df["day"] = df["dep_time"].dt.date
    df["is_weekend"] = df["dep_time"].dt.weekday >= 5
    return df[
        (df["dist_m"] > 0)
        & (df["dur_s"] > 0)
        & (df["dist_km"] < 30)
        & (df["dur_min"] < 180)
    ]


def load_stations() -> pd.DataFrame:
    st = pd.read_csv(ensure_stations())
    return st[["Nimi", "x", "y", "Kapasiteet"]].rename(
        columns={"Nimi": "station", "x": "lon", "y": "lat", "Kapasiteet": "capacity"}
    )


app = Dash(
    __name__,
    title="赫尔辛基共享单车 · 开放数据演示",
)

app.layout = html.Div(
    style={"fontFamily": "Microsoft YaHei, sans-serif", "maxWidth": "1100px",
           "margin": "0 auto", "padding": "24px"},
    children=[
        html.H2("赫尔辛基 / 埃斯波共享单车行程 · 开放数据交互演示", style={"color": BRAND}),
        html.P(
            [
                "数据源：HSL，经芬兰国家开放数据门户 ",
                html.A("avoindata.suomi.fi", href="https://avoindata.suomi.fi", target="_blank"),
                " 发布（CC BY 4.0）。选择月份与站点后，图表联动更新。",
            ],
            style={"color": "#555"},
        ),
        html.Div(
            style={"display": "flex", "gap": "24px", "alignItems": "center", "flexWrap": "wrap"},
            children=[
                html.Div(
                    children=[
                        html.Label("月份"),
                        dcc.Dropdown(
                            id="month",
                            options=[{"label": m, "value": m} for m in MONTHS],
                            value="2024-06",
                            clearable=False,
                            style={"width": "140px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("站点筛选（可选，按借/还任意端匹配）"),
                        dcc.Dropdown(
                            id="station",
                            options=[],
                            placeholder="全部站点",
                            style={"width": "340px"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Label("条数上限 Top N"),
                        dcc.Slider(id="topn", min=5, max=30, step=5, value=15,
                                   marks={5: "5", 15: "15", 30: "30"}),
                    ],
                    style={"width": "200px"},
                ),
            ],
        ),
        html.Hr(),
        html.Div(
            id="kpis",
            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
        ),
        dcc.Loading(
            children=[
                dcc.Graph(id="map"),
                dcc.Graph(id="hourly"),
                dcc.Graph(id="duration"),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr"},
                    children=[dcc.Graph(id="topstations"), dcc.Graph(id="daily")],
                ),
            ]
        ),
        html.Footer(
            "教学演示 | 数据 © HSL via avoindata.suomi.fi, CC BY 4.0",
            style={"color": "#999", "fontSize": "12px", "marginTop": "24px"},
        ),
    ],
)


def kpi_card(title: str, value: str, sub: str = "") -> html.Div:
    return html.Div(
        style={
            "background": "#f5f6fa", "borderRadius": "10px", "padding": "14px 20px",
            "minWidth": "160px", "border": "1px solid #e2e4ef",
        },
        children=[
            html.Div(title, style={"fontSize": "13px", "color": "#888"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": "bold", "color": BRAND}),
            html.Div(sub, style={"fontSize": "12px", "color": "#aaa"}),
        ],
    )


@app.callback(
    Output("station", "options"),
    Input("month", "value"),
)
def fill_station_options(month):
    df = load_trips(month)
    names = sorted(set(df["dep_station"]).union(df["ret_station"]))
    return [{"label": n, "value": n} for n in names]


@app.callback(
    Output("kpis", "children"),
    Output("map", "figure"),
    Output("hourly", "figure"),
    Output("duration", "figure"),
    Output("topstations", "figure"),
    Output("daily", "figure"),
    Input("month", "value"),
    Input("station", "value"),
    Input("topn", "value"),
)
def update(month, station, topn):
    df = load_trips(month)
    scope = "全部站点"
    if station:
        df = df[(df["dep_station"] == station) | (df["ret_station"] == station)]
        scope = station

    kpis = [
        kpi_card("有效行程", f"{len(df):,}", scope),
        kpi_card("距离中位数", f"{df.dist_km.median():.2f} km", f"均值 {df.dist_km.mean():.2f} km"),
        kpi_card("时长中位数", f"{df.dur_min.median():.1f} min", f"均值 {df.dur_min.mean():.1f} min"),
        kpi_card("借还同站比例", f"{(df.dep_station == df.ret_station).mean():.1%}"),
    ]

    # 站点地图：借还总量气泡图
    counts = pd.concat([df["dep_station"], df["ret_station"]]).value_counts()
    st = load_stations()
    st["volume"] = st["station"].map(counts).fillna(0)
    map_fig = px.scatter_map(
        st[st["volume"] > 0],
        lat="lat", lon="lon", size="volume", color="volume",
        hover_name="station", hover_data={"volume": True, "capacity": True, "lat": False, "lon": False},
        zoom=10, center={"lat": 60.19, "lon": 24.94},
        color_continuous_scale="OrRd", size_max=24,
        labels={"volume": "借还次数", "capacity": "车桩数"},
        title=f"站点借还量分布（{month}）",
    )
    map_fig.update_layout(map_style="open-street-map", margin=dict(l=0, r=0, t=40, b=0))

    # 出发时刻：工作日 vs 周末
    hh = (
        df.groupby(["is_weekend", "hour"]).size()
        .rename("n").reset_index()
        .replace({"is_weekend": {False: "工作日", True: "周末"}})
    )
    hourly_fig = px.line(
        hh, x="hour", y="n", color="is_weekend",
        color_discrete_map={"工作日": BRAND, "周末": ACCENT},
        labels={"hour": "小时", "n": "行程数", "is_weekend": ""},
        title=f"出发时刻分布：工作日双峰 vs 周末午后峰（{month}）",
    )

    # 时长直方图
    dur_fig = px.histogram(
        df[df["dur_min"] < 40], x="dur_min", nbins=40,
        labels={"dur_min": "时长（分钟）"},
        title="单次行程时长分布（截尾 40 min）",
    )
    dur_fig.add_vline(x=df["dur_min"].median(), line_color=ACCENT, line_width=2)
    dur_fig.update_layout(bargap=0.05, showlegend=False)

    # Top N 站点
    top = counts.head(topn)
    top_fig = px.bar(
        x=top.values[::-1], y=top.index[::-1], orientation="h",
        labels={"x": "行程数", "y": ""},
        title=f"借还总次数 Top{topn} 站点",
        color_discrete_sequence=["#00B48D"],
    )

    # 逐日
    daily = df.groupby("day").size()
    daily_fig = go.Figure(
        go.Scatter(x=daily.index.astype(str), y=daily.values,
                   mode="lines", line={"color": BRAND, "width": 2})
    )
    daily_fig.update_layout(
        title=f"逐日行程数（{month}）", xaxis_title="日期", yaxis_title="行程数"
    )

    return kpis, map_fig, hourly_fig, dur_fig, top_fig, daily_fig


if __name__ == "__main__":
    app.run(debug=False, port=8050)
