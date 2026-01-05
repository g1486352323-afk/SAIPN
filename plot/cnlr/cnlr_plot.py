#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.lines import Line2D

# ================= 颜色配置 =================

# 顶部图表颜色 (保持不变)
COLOR_IMP_DOT  = "#A6D8F8"   # 隐式散点 (浅蓝)
COLOR_IMP_LINE = "#00B0F0"   # 隐式线条 (亮蓝)
COLOR_EXP_DOT  = "#C9985F"   # 显式散点 (棕褐)
COLOR_EXP_LINE = "#FFD700"   # 显式线条 (金黄)

# 底部图表颜色 (已修改)
COLOR_BOX_FILL = "#2424D2"   # 【修改】箱体填充色 (用户指定)
COLOR_BOX_EDGE = "#2424D2"   # 【修改】箱体边框色 (保持一致或略深)

sns.set_theme(style="darkgrid")

# ================= 1. 数据读取与解析 =================

def parse_first_appearance(filepath):
    """
    读取 CSV，解析 top10_percent_nodes 列
    """
    print(f"读取文件: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"警告：找不到文件 {filepath}，正在生成随机测试数据以供预览...")
        return generate_mock_data()

    if "w_value" not in df.columns:
        raise ValueError(f"CSV 中找不到 'w_value' 列: {filepath}")

    df["w_value"] = pd.to_datetime(df["w_value"])
    df = df.sort_values("w_value")

    node_first_date = {}

    for _, row in df.iterrows():
        date = row["w_value"]
        raw_list = row.get("top10_percent_nodes", "[]")
        try:
            node_list = ast.literal_eval(raw_list)
        except Exception:
            continue
        for item in node_list:
            node_id = str(item[0])
            if node_id not in node_first_date:
                node_first_date[node_id] = date

    return node_first_date

def generate_mock_data():
    # 仅用于无文件时的测试
    nodes = [str(i) for i in range(500)]
    dates = pd.date_range(start="2023-01-01", periods=100)
    return {n: np.random.choice(dates) for n in nodes}

# ================= 2. 底部绘图逻辑 (纯箱线图) =================

def plot_bottom_panel(ax, data, x_col, y_col, order):
    """
    绘制干净的箱线图 (Box Plot)
    - 颜色已更新为 #2424D2
    - 不显示散点 (showfliers=False)
    """
    
    sns.boxplot(
        data=data,
        x=x_col,
        y=y_col,
        order=order,
        ax=ax,
        width=0.35,              # 箱子宽度
        color=COLOR_BOX_FILL,    # 填充颜色
        linewidth=1.5,           # 边框粗细
        showfliers=False,        # 不显示异常值点
        
        # 设置箱体样式
        boxprops={
            'edgecolor': COLOR_BOX_EDGE, 
            'alpha': 0.9         # 稍微提高不透明度，让颜色更实
        },
        # 设置中位线样式
        medianprops={
            'color': 'white',    # 中位数用白色，对比明显
            'linewidth': 1.5,
            'alpha': 1.0
        },
        # 设置须线样式
        whiskerprops={'color': COLOR_BOX_EDGE, 'linewidth': 1.5},
        capprops={'color': COLOR_BOX_EDGE, 'linewidth': 1.5},
    )

# ================= 3. 主程序 =================

def plot_top10_intersection(
    yin_path="top10_percent_yinshi.csv",
    xi_path="top10_percent_xianshi.csv",
    output_path="Top10_Blue_Box.png",
    day_tick_interval: int = 1,
):
    # ---- 3.1 数据准备 ----
    try:
        dates_imp = parse_first_appearance(yin_path)
        dates_exp = parse_first_appearance(xi_path)
    except Exception as e:
        print(e)
        return

    common_nodes = set(dates_imp.keys()) & set(dates_exp.keys())
    print(f"共找到 {len(common_nodes)} 个交集节点。")

    if not common_nodes:
        print("没有交集节点，无法绘图。")
        return

    rows = []
    for node in common_nodes:
        d_imp = dates_imp[node]
        d_exp = dates_exp[node]
        # gap < 0 表示隐式更早 (Implicit Advantage)
        gap_days = (d_imp - d_exp).total_seconds() / (24 * 3600)

        rows.append({
            "node_group": "",
            "node_id": node,
            "xianshi_date": d_exp,
            "yinshi_date": d_imp,
            "days_advantage": gap_days,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("yinshi_date")

    # 分组
    df["node_group"] = pd.qcut(
        np.arange(len(df)), 5, labels=[f"Group {i+1}" for i in range(5)]
    )

    # 顶部图数据转换
    plot_data = []
    for _, row in df.iterrows():
        plot_data.append({"node_group": row["node_group"], "date": row["yinshi_date"], "network": "Implicit Network"})
        plot_data.append({"node_group": row["node_group"], "date": row["xianshi_date"], "network": "Explicit Network"})
    
    df_plot = pd.DataFrame(plot_data)
    group_order = df["node_group"].cat.categories
    df_plot["node_group"] = pd.Categorical(df_plot["node_group"], categories=group_order, ordered=True)

    # ---- 3.2 画布布局 ----
    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # ---- 3.3 顶部绘图 (保持原样) ----
    
    # 散点
    sns.stripplot(data=df_plot[df_plot["network"] == "Implicit Network"], x="node_group", y="date", ax=ax1,
                  color=COLOR_IMP_DOT, marker="D", size=6, jitter=0.25, edgecolor="white", linewidth=0.5, alpha=0.8, zorder=2)
    sns.stripplot(data=df_plot[df_plot["network"] == "Explicit Network"], x="node_group", y="date", ax=ax1,
                  color=COLOR_EXP_DOT, marker="D", size=6, jitter=0.25, edgecolor="white", linewidth=0.5, alpha=0.8, zorder=3)

    # 中位线
    summary = df_plot.groupby(["node_group", "network"], observed=True)["date"].median().reset_index()
    group_map = {g: i for i, g in enumerate(group_order)}
    summary["x_pos"] = summary["node_group"].map(group_map)

    # 连线
    ax1.plot(summary[summary["network"]=="Implicit Network"]["x_pos"], summary[summary["network"]=="Implicit Network"]["date"],
             color=COLOR_IMP_LINE, linewidth=3, marker="x", markersize=8, mew=2, zorder=4)
    ax1.plot(summary[summary["network"]=="Explicit Network"]["x_pos"], summary[summary["network"]=="Explicit Network"]["date"],
             color=COLOR_EXP_LINE, linewidth=3, marker="D", markersize=6, mec="white", zorder=5)

    # 顶部轴设置
    ax1.yaxis.set_major_locator(mdates.DayLocator(interval=max(1, int(day_tick_interval))))
    ax1.yaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax1.set_ylabel("Detection Date", fontsize=12, fontweight='bold')
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", labelbottom=False)
    
    # 图例
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', label='Explicit Network', markerfacecolor=COLOR_EXP_DOT, markersize=10),
        Line2D([0], [0], marker='D', color='w', label='Implicit Network', markerfacecolor=COLOR_IMP_DOT, markersize=10)
    ]
    ax1.legend(handles=legend_elements, loc="upper left", ncol=1, frameon=True)

    # ---- 3.4 底部绘图 (应用新颜色) ----
    
    plot_bottom_panel(ax2, df, "node_group", "days_advantage", group_order)

    # 0刻度线
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6)
    
    # 底部轴设置
    ax2.set_ylabel("Lag Days\n(Negative = Implicit Earlier)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Node Groups (Sorted by Time)", fontsize=12)
    ax2.set_xticks(range(len(group_order)))
    ax2.set_xticklabels(group_order, rotation=0, fontsize=11)
    
    # 调整 Y 轴范围
    y_min, y_max = df["days_advantage"].min(), df["days_advantage"].max()
    ax2.set_ylim(y_min * 1.1, max(5, y_max + 5))

    # ---- 保存 ----
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.1, right=0.95)
    print(f"正在保存图片至: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def _generate_top10_csvs_from_snapshots(
    implicit_dir: str,
    explicit_dir: str,
    output_dir: str,
    topk: float,
    alpha: float,
    max_iter: int,
) -> tuple[str, str]:
    try:
        import generate_compare_inputs_daily as gen
    except Exception as e:
        raise RuntimeError("Failed to import generate_compare_inputs_daily.py in the same folder") from e

    os.makedirs(output_dir, exist_ok=True)

    argv_bak = sys.argv
    sys.argv = [
        "generate_compare_inputs_daily.py",
        "--implicit-dir",
        implicit_dir,
        "--explicit-dir",
        explicit_dir,
        "--output-dir",
        output_dir,
        "--topk",
        str(topk),
        "--alpha",
        str(alpha),
        "--max-iter",
        str(max_iter),
    ]
    try:
        gen.main()
    finally:
        sys.argv = argv_bak

    yin_path = os.path.join(output_dir, "top10_percent_yinshi.csv")
    xi_path = os.path.join(output_dir, "top10_percent_xianshi.csv")
    return yin_path, xi_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--implicit-dir", type=str, default=None)
    ap.add_argument("--explicit-dir", type=str, default=None)
    ap.add_argument("--yin-csv", type=str, default=None)
    ap.add_argument("--xi-csv", type=str, default=None)
    ap.add_argument("--output", type=str, default="Top10_Blue_Box.png")
    ap.add_argument("--work-dir", type=str, default=".")

    ap.add_argument("--topk", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--day-tick-interval", type=int, default=1)

    args = ap.parse_args()

    if args.implicit_dir and args.explicit_dir:
        yin_csv, xi_csv = _generate_top10_csvs_from_snapshots(
            implicit_dir=args.implicit_dir,
            explicit_dir=args.explicit_dir,
            output_dir=args.work_dir,
            topk=args.topk,
            alpha=args.alpha,
            max_iter=args.max_iter,
        )
    else:
        if not args.yin_csv or not args.xi_csv:
            raise SystemExit(
                "Need either (--implicit-dir and --explicit-dir) or (--yin-csv and --xi-csv)."
            )
        yin_csv, xi_csv = args.yin_csv, args.xi_csv

    plot_top10_intersection(
        yin_path=yin_csv,
        xi_path=xi_csv,
        output_path=args.output,
        day_tick_interval=args.day_tick_interval,
    )

if __name__ == "__main__":
    main()