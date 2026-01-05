import os
import glob
import re
import argparse
import networkx as nx
import importlib
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# ================= 配置区域 =================
# 默认使用 PHEME 目录下 30min 粒度的显式/隐式输出
EXPLICIT_DIR = '/data_huawei/gaohaizhen/network/saipn/model/PHEME/output/explicit-30min'
IMPLICIT_DIR = '/data_huawei/gaohaizhen/network/saipn/model/PHEME/output/implicit-ablation-30min'
TOP_K_RATIO = 0.10  # 使用 Top10%
# 支持的中心性：pagerank / indegree / katz
DEFAULT_METRIC = 'indegree'
# Katz 后端：cpu（nx 迭代版）或 gpu（cugraph）
DEFAULT_KATZ_BACKEND = 'gpu'
# ===========================================

def parse_timestamp(filename):
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2})", filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d_%H-%M')
    match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{4})", filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d_%H%M')
    # Metaverse 显式快照：explicit-YYYY-MM-DD.edgelist（只有日期，没有时间）
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    if match:
        return datetime.strptime(match.group(1), '%Y-%m-%d')
    return None

def collect_files(directory):
    """收集目录内所有快照文件并解析时间戳。"""
    files = sorted(glob.glob(os.path.join(directory, "*.edgelist")))
    if not files:
        files = sorted(glob.glob(os.path.join(directory, 'snapshots', "*.edgelist")))
    files_with_ts = []
    for f in files:
        ts = parse_timestamp(os.path.basename(f))
        if ts:
            files_with_ts.append((ts, f))
    files_with_ts.sort(key=lambda x: x[0])
    return files_with_ts


def compute_scores(G: nx.DiGraph, metric: str, katz_backend: str):
    """根据 metric 计算节点得分."""
    metric = metric.lower()
    if metric == 'indegree':
        # 带权入度
        return {n: float(G.in_degree(n, weight='weight')) for n in G.nodes()}
    if metric == 'katz':
        if katz_backend == 'gpu':
            # 尝试使用 CuGraph（稀疏 GPU 版）
            try:
                cudf = importlib.import_module("cudf")
                cugraph = importlib.import_module("cugraph")
                if G.number_of_nodes() == 0:
                    return {}
                # 组装边表
                rows = []
                for u, v, d in G.edges(data=True):
                    w = d.get('weight', 1.0)
                    rows.append((u, v, float(w)))
                if not rows:
                    return {}
                pdf = pd.DataFrame(rows, columns=['src', 'dst', 'weight'])
                gdf = cudf.DataFrame.from_pandas(pdf)
                g = cugraph.DiGraph()
                g.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='weight', renumber=False)
                res = cugraph.katz_centrality(g, alpha=0.001, beta=1.0, max_iter=200, tol=1e-4)
                # res: cudf with ['vertex','katz_centrality']
                s = res.to_pandas()
                return dict(zip(s['vertex'].astype(str), s['katz_centrality'].astype(float)))
            except Exception as e:
                print(f"[Warn] CuGraph katz failed, fallback to CPU: {e}")
                # 回退到 CPU 迭代版
        # CPU 迭代版（稀疏）
        try:
            return nx.katz_centrality(G, alpha=0.001, beta=1.0, weight='weight', max_iter=200, tol=1e-4)
        except Exception:
            return nx.katz_centrality(G, alpha=0.001, beta=1.0, max_iter=200, tol=1e-4)
    # 默认 PageRank
    try:
        return nx.pagerank(G, weight='weight')
    except Exception:
        return nx.pagerank(G)


def dynamic_replay(files_with_ts, metric: str, katz_backend: str, is_explicit=True):
    """
    按时间累积图，记录首次进入 TopK 的时间。
    返回：burst_times(dict: node->datetime), t0(最早时间)
    """
    if not files_with_ts:
        return {}, None

    G = nx.DiGraph()
    burst_times = {}
    birth_times = {}

    for ts, f in files_with_ts:
        try:
            if is_explicit:
                G_snap = nx.read_edgelist(
                    f, data=(('weight', float), ('type', str)), create_using=nx.DiGraph()
                )
            else:
                G_snap = nx.read_edgelist(
                    f, data=(('weight', float),), create_using=nx.DiGraph()
                )
            G.add_edges_from(G_snap.edges(data=True))

            # 记录首次出现时间
            for n in G_snap.nodes():
                if n not in birth_times:
                    birth_times[n] = ts
        except Exception as e:
            print(f"[Warn] Fail to load {f}: {e}")
            continue

        if len(G) < 10:
            continue

        scores = compute_scores(G, metric, katz_backend)
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        k = max(1, int(len(sorted_nodes) * TOP_K_RATIO))
        top_nodes = {n for n, _ in sorted_nodes[:k]}

        for node in top_nodes:
            # 延迟判定：节点出现当步不计核心，至少过1个时间步
            if birth_times.get(node) == ts:
                continue
            if node not in burst_times:
                burst_times[node] = ts

    return burst_times, files_with_ts[0][0]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--explicit-dir', default=EXPLICIT_DIR)
    ap.add_argument('--implicit-dir', default=IMPLICIT_DIR)
    ap.add_argument('--metric', default=DEFAULT_METRIC, choices=['pagerank', 'indegree', 'katz'])
    ap.add_argument('--katz-backend', default=DEFAULT_KATZ_BACKEND, choices=['cpu', 'gpu'])
    return ap.parse_args()


def main():
    args = parse_args()
    explicit_dir = args.explicit_dir
    implicit_dir = args.implicit_dir
    metric = args.metric
    katz_backend = args.katz_backend

    # 1. 收集文件并对齐时间轴（取交集）
    exp_files = collect_files(explicit_dir)
    imp_files = collect_files(implicit_dir)

    if not exp_files or not imp_files:
        print("[Error] 显式或隐式快照为空，无法计算 CNLR。")
        return

    exp_map = {ts: f for ts, f in exp_files}
    imp_map = {ts: f for ts, f in imp_files}
    common_ts = sorted(set(exp_map.keys()) & set(imp_map.keys()))

    if not common_ts:
        print("[Error] 显式/隐式时间戳无交集，无法计算。")
        return

    # 按交集时间轴重放
    exp_files_aligned = [(ts, exp_map[ts]) for ts in common_ts]
    imp_files_aligned = [(ts, imp_map[ts]) for ts in common_ts]

    print(f"[Info] 对齐后的时间点数: {len(common_ts)} (取交集)")
    print(f"  显式首/末时间: {exp_files_aligned[0][0]} -> {exp_files_aligned[-1][0]}")
    print(f"  隐式首/末时间: {imp_files_aligned[0][0]} -> {imp_files_aligned[-1][0]}")

    exp_core, exp_t0 = dynamic_replay(exp_files_aligned, metric=metric, katz_backend=katz_backend, is_explicit=True)
    imp_core, imp_t0 = dynamic_replay(imp_files_aligned, metric=metric, katz_backend=katz_backend, is_explicit=False)

    if not exp_core or not imp_core:
        print("[Error] 无法得到核心进入时间，检查网络是否过于稀疏。")
        return

    t_start = min(exp_t0, imp_t0)

    print("\n" + "="*60)
    print("🔍 DETAILED ANALYSIS REPORT")
    print("="*60)
    
    details = []
    
    for node, t_exp_core in exp_core.items():
        if node not in imp_core:
            continue
        t_imp_core = imp_core[node]

        diff_hours = (t_exp_core - t_imp_core).total_seconds() / 3600.0
        duration_exp_hours = (t_exp_core - t_start).total_seconds() / 3600.0
        if duration_exp_hours <= 0:
            continue

        cnlr_u = diff_hours / duration_exp_hours

        category = "DRAW"
        if diff_hours > 0:
            category = "WIN (Early)"
        elif diff_hours < 0:
            category = "LOSS (Late)"

        details.append({
            'node': node,
            't_start': t_start,
            't_exp_core': t_exp_core,
            't_imp_core': t_imp_core,
            'diff_hours': diff_hours,
            'duration_exp_hours': duration_exp_hours,
            'cnlr': cnlr_u,
            'category': category
        })
            
    df = pd.DataFrame(details)
    
    if df.empty:
        print("No overlap found.")
        return

    out_dir = os.path.dirname(os.path.abspath(__file__))
    detailed_path = os.path.join(out_dir, 'cnlr_detailed.csv')
    df.to_csv(detailed_path, index=False)

    # --- 打印统计 ---
    print(f"Total Overlap Nodes: {len(df)}")
    print(f"Win: {len(df[df['category'] == 'WIN (Early)'])}")
    print(f"Loss: {len(df[df['category'] == 'LOSS (Late)'])}")
    print(f"Draw: {len(df[df['category'] == 'DRAW'])}")
    
    # --- 诊断 1: 检查“平局”的具体时间 ---
    print("\n[Diagnosis 1] Inspecting 'DRAW' cases (Sample 5):")
    draws = df[df['category'] == 'DRAW'].head(5)
    if not draws.empty:
        print(draws[['node', 't_exp_core', 't_imp_core', 'diff_hours']].to_string(index=False))
    else:
        print("  No draws.")
        
    # --- 诊断 2: 检查“赢”的具体时间 ---
    print("\n[Diagnosis 2] Inspecting 'WIN' cases (Sample 5):")
    wins = df[df['category'] == 'WIN (Early)'].head(5)
    if not wins.empty:
        print(wins[['node', 't_exp_core', 't_imp_core', 'diff_hours']].to_string(index=False))
    else:
        print("  No wins.")

    summary = {
        'total_overlap_nodes': len(df),
        'win_count': int((df['category'] == 'WIN (Early)').sum()),
        'loss_count': int((df['category'] == 'LOSS (Late)').sum()),
        'draw_count': int((df['category'] == 'DRAW').sum()),
        'mean_cnlr': float(df['cnlr'].mean()),
        'median_cnlr': float(df['cnlr'].median()),
        'std_cnlr': float(df['cnlr'].std(ddof=0)),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(out_dir, 'cnlr_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("💡 结论与建议")
    
    avg_gap = 0
    if len(exp_files) > 1:
         avg_gap = (exp_files[1][0] - exp_files[0][0]).total_seconds()/3600

    if avg_gap >= 6.0:
        print(f"1. 你的时间粒度是 {avg_gap} 小时。")
        print("   这意味着发生在 12:00 到 18:00 之间的所有变化都被压扁在同一个时间戳上了。")
        print("   显式和隐式大概率落在同一个6小时窗口里，导致 Diff=0。")
        print("   👉 建议：必须把数据切分得更细（如 15分钟 或 1小时）。")

if __name__ == '__main__':
    main()