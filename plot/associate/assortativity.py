#!/usr/bin/env python

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def _extract_snapshot_datetime(path: Path) -> Optional[pd.Timestamp]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2})", path.name)
    if m:
        return pd.to_datetime(f"{m.group(1)} {m.group(2).replace('-', ':')}", errors="coerce")

    m = re.search(r"(\d{4}-\d{2}-\d{2})(?=\.edgelist$)", path.name)
    if m:
        return pd.to_datetime(m.group(1), errors="coerce")

    return None


def _iter_edges_from_edgelist(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            s = s.replace("→", " ").replace("->", " ").replace("-->", " ")
            toks = s.split()
            if len(toks) < 2:
                continue
            yield toks[0], toks[1]


def _degree_assortativity_for_snapshot(path: Path, as_undirected: bool = True) -> float:
    if as_undirected:
        G: nx.Graph = nx.Graph()
    else:
        G = nx.DiGraph()

    for u, v in _iter_edges_from_edgelist(path):
        G.add_edge(u, v)

    if G.number_of_edges() < 2:
        return float("nan")

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if as_undirected:
                val = nx.degree_assortativity_coefficient(G)
            else:
                val = nx.degree_assortativity_coefficient(G, x="out", y="in")
    except Exception:
        return float("nan")

    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return float("nan")
    return float(val)


def find_snapshot_files(snapshots_dir: Path) -> List[Path]:
    patterns = [
        "implicit_edges_*.edgelist",
        "tweet_network-*.edgelist",
        "explicit-*.edgelist",
    ]
    files: List[Path] = []
    for pat in patterns:
        files.extend(list(snapshots_dir.glob(pat)))
    files = [p for p in files if p.is_file() and p.stat().st_size > 0]
    return files


def build_assortativity_timeseries(
    snapshots_dir: Path,
    as_undirected: bool = True,
    max_files: Optional[int] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    files = find_snapshot_files(snapshots_dir)
    if not files:
        raise FileNotFoundError(f"No edgelist snapshots found under {snapshots_dir}")

    dated: List[Tuple[pd.Timestamp, Path]] = []
    for p in files:
        ts = _extract_snapshot_datetime(p)
        if ts is None or pd.isna(ts):
            continue
        dated.append((ts, p))

    dated.sort(key=lambda x: x[0])
    if max_files is not None and max_files > 0:
        dated = dated[:max_files]

    if start_date is not None:
        dated = [(ts, p) for ts, p in dated if ts >= start_date]
    if end_date is not None:
        dated = [(ts, p) for ts, p in dated if ts <= end_date]

    rows = []
    for ts, p in dated:
        val = _degree_assortativity_for_snapshot(p, as_undirected=as_undirected)
        rows.append((ts, val))

    df = pd.DataFrame(rows, columns=["date", "degree_assortativity"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def plot_assortativity(df: pd.DataFrame, out_png: Path) -> None:
    if df.empty:
        raise ValueError("Empty assortativity dataframe; nothing to plot")

    df_plot = df.copy()
    df_plot["degree_assortativity"] = pd.to_numeric(df_plot["degree_assortativity"], errors="coerce")
    df_plot.loc[~np.isfinite(df_plot["degree_assortativity"]), "degree_assortativity"] = np.nan
    if not np.isfinite(df_plot["degree_assortativity"]).any():
        raise ValueError("All assortativity values are NaN/Inf; nothing to plot")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    # === 【新增】 绘制零线生命线 (The Zero Line) ===
    # zorder=1 保证它在数据点（zorder 2 和 3）的下方，但在网格线的上方
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

    # === 颜色设定 ===
    main_color = "#1f77b4"

    # === 绘制折线 ===
    ax.plot(
        df_plot["date"],
        df_plot["degree_assortativity"],
        color=main_color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
        zorder=2,
    )
    
    finite_mask = np.isfinite(df_plot["degree_assortativity"].to_numpy(dtype=float))
    
    # === 绘制散点 ===
    ax.scatter(
        df_plot.loc[finite_mask, "date"],
        df_plot.loc[finite_mask, "degree_assortativity"],
        s=30,
        marker="o",
        facecolor=main_color,
        edgecolor="none",
        linewidth=0.0,
        alpha=0.65,
        zorder=3,
    )

    ax.set_ylabel("Degree assortativity")
    ax.set_xlabel("")

    # === 根据文件名区分 Y 轴刻度策略 ===
    if "explicit" in out_png.name:
        # === 针对 Explicit 的定制刻度 ===
        y_min_limit, y_max_limit = -0.3, 0.4
        yticks = np.arange(y_min_limit, y_max_limit + 1e-9, 0.1)
        ax.set_yticks(yticks)
        ax.set_ylim(y_min_limit, y_max_limit)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        
    else:
        # === 针对 Implicit 的原有逻辑 ===
        y_vals = df_plot["degree_assortativity"].astype(float).to_numpy()
        y_min = float(np.nanmin(y_vals))
        y_max = float(np.nanmax(y_vals))
        
        y0 = np.floor(y_min * 10.0) / 10.0
        y1 = np.ceil(y_max * 10.0) / 10.0
        
        if y1 < 0.7:
            y1 = 0.7
            
        yticks = np.arange(y0, y1 + 1e-9, 0.1)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        
        pad = 0.02
        ax.set_ylim(y0 - pad, y1 + pad)

    # X轴时间格式
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # === 刻度线样式 ===
    ax.tick_params(
        axis='both', 
        which='major', 
        direction='out', 
        length=6, 
        width=1.2, 
        colors='black', 
        grid_alpha=0.5,
        bottom=True,
        left=True
    )

    # 图例
    legend_handle = Line2D(
        [0],
        [0],
        color=main_color,
        linestyle="--",
        linewidth=1.0,
        marker="o",
        markersize=6,
        markerfacecolor=main_color,
        markeredgecolor="none",
        alpha=0.8,
        label="degree_assortativity",
    )
    ax.legend(handles=[legend_handle], loc="upper left", frameon=True)
    
    # 网格和边框
    ax.grid(True, which="major", axis='y', color="lightgrey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(False, axis='x')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_assortativity_compare(df_implicit: pd.DataFrame, df_explicit: pd.DataFrame, out_png: Path) -> None:
    if df_implicit.empty and df_explicit.empty:
        raise ValueError("Empty assortativity dataframes; nothing to plot")

    df_i = df_implicit.copy() if not df_implicit.empty else pd.DataFrame(columns=["date", "degree_assortativity"])
    df_e = df_explicit.copy() if not df_explicit.empty else pd.DataFrame(columns=["date", "degree_assortativity"])

    for df in (df_i, df_e):
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["degree_assortativity"] = pd.to_numeric(df["degree_assortativity"], errors="coerce")
            df.loc[~np.isfinite(df["degree_assortativity"]), "degree_assortativity"] = np.nan
            df.sort_values("date", inplace=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=120)

    # === 【新增】 绘制零线生命线 (The Zero Line) ===
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)

    implicit_color = "#1f77b4"
    explicit_color = "#ff7f0e"

    if not df_i.empty:
        finite_mask_i = np.isfinite(df_i["degree_assortativity"].to_numpy(dtype=float))
        ax.plot(
            df_i["date"],
            df_i["degree_assortativity"],
            color=implicit_color,
            linestyle="-",
            linewidth=1.2,
            alpha=0.6,
            zorder=2,
        )
        ax.scatter(
            df_i.loc[finite_mask_i, "date"],
            df_i.loc[finite_mask_i, "degree_assortativity"],
            s=28,
            marker="o",
            facecolor=implicit_color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.75,
            zorder=3,
        )

    if not df_e.empty:
        finite_mask_e = np.isfinite(df_e["degree_assortativity"].to_numpy(dtype=float))
        ax.plot(
            df_e["date"],
            df_e["degree_assortativity"],
            color=explicit_color,
            linestyle="-",
            linewidth=1.2,
            alpha=0.6,
            zorder=2,
        )
        ax.scatter(
            df_e.loc[finite_mask_e, "date"],
            df_e.loc[finite_mask_e, "degree_assortativity"],
            s=28,
            marker="o",
            facecolor=explicit_color,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.75,
            zorder=3,
        )

    ax.set_ylabel("Degree assortativity")
    ax.set_xlabel("")

    y_min_limit, y_max_limit = -0.3, 0.7
    yticks = np.arange(y_min_limit, y_max_limit + 1e-9, 0.1)
    ax.set_yticks(yticks)
    ax.set_ylim(y_min_limit, y_max_limit)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1.2,
        colors="black",
        grid_alpha=0.5,
        bottom=True,
        left=True,
    )

    handles: List[Line2D] = []
    if not df_i.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                color=implicit_color,
                linestyle="-",
                linewidth=1.2,
                marker="o",
                markersize=6,
                markerfacecolor=implicit_color,
                markeredgecolor="none",
                alpha=0.9,
                label="implicit",
            )
        )
    if not df_e.empty:
        handles.append(
            Line2D(
                [0],
                [0],
                color=explicit_color,
                linestyle="-",
                linewidth=1.2,
                marker="o",
                markersize=6,
                markerfacecolor=explicit_color,
                markeredgecolor="none",
                alpha=0.9,
                label="explicit",
            )
        )
    if handles:
        ax.legend(handles=handles, loc="upper left", frameon=True)

    ax.grid(True, which="major", axis="y", color="lightgrey", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.grid(False, axis="x")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--implicit-dir",
        type=str,
        default="/data_huawei/gaohaizhen/network/saipn/model/metaverse/output/full_best_meta/snapshots",
    )
    ap.add_argument(
        "--explicit-dir",
        type=str,
        default="/data_huawei/gaohaizhen/network/saipn/model/metaverse/output/explicit_metaverse_best/snapshots",
    )
    ap.add_argument("--output-dir", type=str, default=".")
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--start-date", type=str, default="2021-10-01")
    ap.add_argument("--end-date", type=str, default="2022-07-31")
    ap.add_argument("--full-range", action="store_true")
    ap.add_argument("--gap-days", type=int, default=30)
    return ap.parse_args()


def _run_one(
    label: str,
    snapshots_dir: Path,
    out_dir: Path,
    directed: bool,
    max_files: Optional[int],
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    gap_days: int,
) -> pd.DataFrame:
    df = build_assortativity_timeseries(
        snapshots_dir=snapshots_dir,
        as_undirected=not directed,
        max_files=max_files,
        start_date=start_date,
        end_date=end_date,
    )

    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
        df["degree_assortativity"] = pd.to_numeric(df["degree_assortativity"], errors="coerce")
        dt = df["date"].diff().dt.total_seconds().div(86400.0)
        gap_mask = dt.gt(float(gap_days))
        if gap_mask.any():
            df.loc[gap_mask, "degree_assortativity"] = np.nan

    out_csv = out_dir / f"assortativity_{label}.csv"
    df.to_csv(out_csv, index=False)
    print(f"[saved] assortativity csv -> {out_csv}")

    out_png = out_dir / f"assortativity_{label}.png"
    plot_assortativity(df, out_png)
    print(f"[saved] assortativity plot -> {out_png}")

    return df


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.full_range:
        start_date = None
        end_date = None
    else:
        start_date = pd.to_datetime(args.start_date, errors="coerce") if args.start_date else None
        if start_date is not None and pd.isna(start_date):
            start_date = None
        end_date = pd.to_datetime(args.end_date, errors="coerce") if args.end_date else None
        if end_date is not None and pd.isna(end_date):
            end_date = None

    implicit_dir = Path(args.implicit_dir)
    explicit_dir = Path(args.explicit_dir)

    df_implicit = pd.DataFrame(columns=["date", "degree_assortativity"])
    df_explicit = pd.DataFrame(columns=["date", "degree_assortativity"])

    if implicit_dir.exists():
        df_implicit = _run_one(
            "implicit",
            implicit_dir,
            out_dir,
            directed=args.directed,
            max_files=args.max_files,
            start_date=start_date,
            end_date=end_date,
            gap_days=args.gap_days,
        )
    else:
        print(f"[warn] implicit dir not found: {implicit_dir}")

    if explicit_dir.exists():
        df_explicit = _run_one(
            "explicit",
            explicit_dir,
            out_dir,
            directed=args.directed,
            max_files=args.max_files,
            start_date=start_date,
            end_date=end_date,
            gap_days=args.gap_days,
        )
    else:
        print(f"[warn] explicit dir not found: {explicit_dir}")

    if (not df_implicit.empty) or (not df_explicit.empty):
        out_png = out_dir / "assortativity_implicit_vs_explicit.png"
        plot_assortativity_compare(df_implicit, df_explicit, out_png)
        print(f"[saved] assortativity compare plot -> {out_png}")


if __name__ == "__main__":
    main()