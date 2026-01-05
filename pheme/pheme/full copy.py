import pandas as pd
import os
import networkx as nx
import datetime
import torch
import numpy as np
from tqdm import tqdm
import csv 
import argparse
import sys
import time
import warnings

# ================= 1. 屏蔽干扰信息 =================
warnings.filterwarnings("ignore")
try:
    # 屏蔽 cuGraph 的性能提示警告
    warnings.filterwarnings("ignore", category=UserWarning, module="cugraph")
except: pass

# ================= 2. 环境配置 =================
try:
    import cugraph
    import cudf
    HAS_GPU_GRAPH = True
    print("[Success] RAPIDS cuGraph Loaded. GPU Acceleration Enabled.")
except ImportError:
    HAS_GPU_GRAPH = False
    print("[Warning] cuGraph not found! Using NetworkX (CPU). Speed will be slow.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {device}")

# 路径配置
BASE_DIR = '/data_huawei/gaohaizhen/network/saipn/model/ablation'
NEW_VECTOR_PATH = os.path.join(BASE_DIR, 'embeddin', 'output_vectors.txt')
NEW_SENTIMENT_PATH = os.path.join(BASE_DIR, 'embeddin', 'final_with_sentiment.csv')
EMBEDDIN_ROOT = os.path.join(BASE_DIR, 'embeddin')

if EMBEDDIN_ROOT not in sys.path: sys.path.append(EMBEDDIN_ROOT)

try:
    from baseline_propagation.metrics import calculate_all_dcprr_scores
    DCPRR_IMPORTED = True
except ImportError:
    DCPRR_IMPORTED = False
    def calculate_all_dcprr_scores(*args, **kwargs): return {}

# ================= 3. 辅助函数 =================

def load_vectors_to_gpu(file_path):
    if not os.path.exists(file_path): return {}
    vector_dict = {}
    print(f"Loading vectors: {file_path} ...")
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            try:
                vec = torch.tensor([float(x) for x in parts[1].split()], dtype=torch.float16, device=device) 
                vector_dict[parts[0].strip()] = vec
            except: continue
    return vector_dict

def calculate_cnlr_fast(current_edges_set, prev_edges_set, communities_dict):
    if not communities_dict or not current_edges_set: return 0.0
    new_edges = current_edges_set - prev_edges_set
    if not new_edges: return 0.0
    
    # 性能保护
    if len(new_edges) > 20000: return 0.0 

    cnlr_scores = []
    for comm_id, nodes in communities_dict.items():
        if len(nodes) < 3: continue
        node_set = set(nodes)
        possible = len(node_set) * (len(node_set) - 1)
        if possible <= 0: continue
        
        internal_new = sum(1 for u, v in new_edges if u in node_set and v in node_set)
        cnlr_scores.append(internal_new / possible)
    
    val = float(np.mean(cnlr_scores)) if cnlr_scores else 0.0
    return 0.0 if np.isnan(val) else val

# ================= 4. 主逻辑 =================

def main_gpu(args):
    start_time = time.time()
    
    OUTPUT_DIR = args.output_dir if args.output_dir else os.path.join(BASE_DIR, 'outputs', 'implicit-ablation')
    if os.path.exists(OUTPUT_DIR):
        for f in ["index_gpu.csv"]:
            p = os.path.join(OUTPUT_DIR, f)
            if os.path.exists(p): os.remove(p)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 加载向量与情感 ---
    vec_path = args.vector_file if args.vector_file else NEW_VECTOR_PATH
    vector_dict = load_vectors_to_gpu(vec_path)
    
    senti_path = args.sentiment_file if args.sentiment_file else NEW_SENTIMENT_PATH
    senti_df = pd.read_csv(senti_path)
    id_col = next((c for c in senti_df.columns if 'id' in c and 'str' in c), 'id_str')
    senti_map = {}
    for _, row in senti_df.iterrows():
        pos, neu, neg = row['positive_probability'], row['neutral_probability'], row['negative_probability']
        val = (1-neu) if pos > neg else -(1-neu)
        senti_map[str(row[id_col])] = val

    # --- 加载主数据 ---
    data_file = args.data_file or os.path.join(BASE_DIR, "charliehebdo_gemini_2_flash_output_fixed_from_cleaned.csv")
    print(f"Loading Data: {data_file}")
    table = pd.read_csv(data_file, dtype={'raw_value.id_str': str})
    table['raw_value.created_at'] = pd.to_datetime(table['raw_value.created_at'])
    
    # 排序并设置索引
    table.sort_values('raw_value.created_at', inplace=True)
    table.set_index('raw_value.created_at', inplace=True) # 这里不需要 drop=False 了，因为我们直接用索引
    
    G = nx.DiGraph()
    MAX_NODES = 80000 
    history_tensor = torch.zeros((MAX_NODES, 385), dtype=torch.float16, device=device)
    history_times = torch.zeros(MAX_NODES, dtype=torch.float32, device=device)
    history_emotions = torch.zeros(MAX_NODES, dtype=torch.float16, device=device)
    history_ids = []  
    history_count = 0
    
    prev_edges_snapshot = set()
    all_stats_records = [] 

    f_csv = open(os.path.join(OUTPUT_DIR, "index_gpu.csv"), 'w', newline='')
    writer = csv.writer(f_csv)
    writer.writerow(['Time', 'Nodes', 'Edges', 'AvgPageRank', 'Assortativity', 'Modularity', 'DCPRR', 'CNLR', 'CompIntensity', 'PropScope', 'CollabIntensity'])
    
    groups = list(table.resample(args.resample))
    print(f"[Info] Total Steps: {len(groups)} (Resample: {args.resample})")
    print(f"[Info] Mode: High Precision (Calculating metrics EVERY step)")

    # --- 循环 ---
    for loop_idx, (w, w_t) in enumerate(tqdm(groups, desc="Simulating")):
        if len(w_t) == 0: continue
        
        current_ts = w.timestamp()
        cutoff_ts = current_ts - (args.delete_after_hours * 3600)
        
        # 1. 节点过期
        if G.number_of_nodes() > 0:
            nodes_to_remove = [n for n, t in G.nodes(data='time') if t is not None and t < cutoff_ts]
            if nodes_to_remove: G.remove_nodes_from(nodes_to_remove)

        # 2. 新节点
        batch_ids, batch_vecs, batch_times, batch_emos = [], [], [], []
        
        # [关键修复]: 直接从 iterrows 的索引(row_index)获取时间，不查字典，解决 KeyError
        for row_index, row in w_t.iterrows():
            tid = str(row['raw_value.id_str'])
            ts = row_index.timestamp() # <--- 这里是修复点，row_index 就是时间
            
            emo = senti_map.get(tid, 0.0)
            if tid in vector_dict:
                b_vec = vector_dict[tid]
                e_tensor = torch.tensor([emo], device=device, dtype=torch.float16)
                comb = torch.cat((b_vec, e_tensor))
                norm = torch.norm(comb)
                if norm > 0: comb /= norm
                batch_ids.append(tid)
                batch_vecs.append(comb)
                batch_times.append(ts)
                batch_emos.append(emo)
                G.add_node(tid, sentiment=float(emo), time=ts)
        
        if not batch_ids: continue

        # 3. 连边计算
        batch_tensor = torch.stack(batch_vecs)
        batch_time_tensor = torch.tensor(batch_times, device=device)
        
        weights_hist, h_mask = None, None
        if history_count > 0:
            sim_h = torch.mm(batch_tensor, history_tensor[:history_count].t())
            diff_h = (batch_time_tensor.unsqueeze(1) - history_times[:history_count].unsqueeze(0)) / 3600.0
            scores_h = sim_h * torch.exp(-torch.abs(diff_h)/max(args.decay_unit_hours, 0.01)) * (history_times[:history_count] >= cutoff_ts).unsqueeze(0)
            h_mask = scores_h > args.score_threshold
            weights_hist = scores_h

        sim_in = torch.mm(batch_tensor, batch_tensor.t())
        scores_in = sim_in * torch.exp(-torch.abs((batch_time_tensor.unsqueeze(1)-batch_time_tensor.unsqueeze(0))/3600.0)/max(args.decay_unit_hours,0.01)) * torch.tril(torch.ones_like(sim_in), -1).bool()
        in_mask = scores_in > args.score_threshold

        batch_final_vecs, batch_final_emos = [], []
        
        # 传播 (CPU)
        for i, tid in enumerate(batch_ids):
            vol = 0.0
            if h_mask is not None and h_mask[i].any():
                w_h = weights_hist[i][h_mask[i]]
                vol += torch.mean(w_h * history_emotions[:history_count][h_mask[i]]).item()
                for k, h_idx in enumerate(torch.nonzero(h_mask[i]).squeeze(1).cpu().tolist()):
                    if G.has_node(history_ids[h_idx]): G.add_edge(history_ids[h_idx], tid, weight=w_h[k].item())
            
            if in_mask[i].any():
                w_in = scores_in[i][in_mask[i]]
                src_idxs = torch.nonzero(in_mask[i]).squeeze(1).cpu().tolist()
                vol += torch.mean(w_in * torch.tensor([batch_emos[x] for x in src_idxs], device=device)).item()
                for k, s_idx in enumerate(src_idxs):
                    G.add_edge(batch_ids[s_idx], tid, weight=w_in[k].item())
            
            new_emo = max(-1.0, min(1.0, batch_emos[i] + vol))
            batch_final_emos.append(new_emo)
            bert_part = batch_tensor[i][:-1]
            new_vec = torch.cat((bert_part, torch.tensor([new_emo], device=device, dtype=torch.float16)))
            batch_final_vecs.append(new_vec / torch.norm(new_vec))
            G.nodes[tid]['sentiment'] = float(new_emo)

        if batch_final_vecs:
            n_new = len(batch_ids)
            if history_count + n_new < MAX_NODES:
                history_tensor[history_count:history_count+n_new] = torch.stack(batch_final_vecs)
                history_times[history_count:history_count+n_new] = batch_time_tensor
                history_emotions[history_count:history_count+n_new] = torch.tensor(batch_final_emos, device=device, dtype=torch.float16)
                history_ids.extend(batch_ids)
                history_count += n_new

        # ================= [指标计算] =================
        num_edges = G.number_of_edges()
        num_nodes = G.number_of_nodes()
        
        pr_val, mod_val, assort_val, dcprr_val, cnlr_val = 0,0,0,0,0
        collab_intensity, comp_intensity, prop_scope = 0,0,0
        communities_dict = {}

        gpu_ready = False
        gdf, G_gpu_und = None, None

        if HAS_GPU_GRAPH and num_edges > 0:
            try:
                node_list = list(G.nodes())
                node_map = {n: i for i, n in enumerate(node_list)}
                edges_data = [(node_map[u], node_map[v], d['weight']) for u, v, d in G.edges(data=True)]
                if edges_data:
                    gdf = cudf.DataFrame(edges_data, columns=['src', 'dst', 'w'])
                    G_gpu = cugraph.Graph(directed=True)
                    G_gpu.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w')
                    G_gpu_und = cugraph.Graph(directed=False)
                    G_gpu_und.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w')
                    gpu_ready = True
            except: pass

        # 1. PageRank
        if gpu_ready:
            try: pr_val = cugraph.pagerank(G_gpu)['pagerank'].mean()
            except: pass
        
        # 2. Louvain
        if gpu_ready:
            try:
                parts, mod_val = cugraph.louvain(G_gpu_und)
                parts_df = parts.to_pandas()
                inv_map = {v: k for k, v in node_map.items()}
                for r in parts_df.itertuples():
                    if r.vertex in inv_map: communities_dict.setdefault(r.partition, []).append(inv_map[r.vertex])
            except: pass
            
        # 3. Assortativity (GPU 修复版)
        if gpu_ready:
            try:
                deg_df = G_gpu_und.degrees()
                cols = [c for c in deg_df.columns if c != 'vertex']
                deg_col = cols[0] if cols else 'degree'
                m = gdf.merge(deg_df, left_on='src', right_on='vertex').rename(columns={deg_col: 'src_deg'})
                m = m[['src_deg', 'dst']].merge(deg_df, left_on='dst', right_on='vertex').rename(columns={deg_col: 'dst_deg'})
                assort_val = m['src_deg'].corr(m['dst_deg'])
                if np.isnan(assort_val): assort_val = 0.0
            except: pass

        # 4. Triangles
        if gpu_ready:
            try: collab_intensity = cugraph.triangle_count(G_gpu_und)['counts'].sum() / 3
            except: pass

        # 5. CNLR
        curr_edges_snapshot = set(G.edges())
        if prev_edges_snapshot:
            cnlr_val = calculate_cnlr_fast(curr_edges_snapshot, prev_edges_snapshot, communities_dict)
        prev_edges_snapshot = curr_edges_snapshot

        # Write
        row = [w.strftime('%Y-%m-%d %H:%M:%S'), num_nodes, num_edges, 
               f"{pr_val:.4f}", f"{assort_val:.4f}", f"{mod_val:.4f}", 
               f"{dcprr_val:.4f}", f"{cnlr_val:.4f}", 
               0, 0, int(collab_intensity)]
        writer.writerow(row)
        f_csv.flush()
        all_stats_records.append([float(x) for x in row[1:]])

    # End
    if all_stats_records:
        avgs = np.mean(np.array(all_stats_records), axis=0)
        writer.writerow(['GLOBAL_AVG'] + [f"{x:.4f}" for x in avgs])
    
    f_csv.close()
    print(f"Done. Time: {time.time()-start_time:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resample', default='H') 
    parser.add_argument('--decay-unit-hours', type=float, default=12.0)
    parser.add_argument('--delete-after-hours', type=float, default=24.0)
    parser.add_argument('--score-threshold', type=float, default=0.70)
    parser.add_argument('--data-file', default=None)
    parser.add_argument('--vector-file', default=None)
    parser.add_argument('--sentiment-file', default=None)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    main_gpu(args)