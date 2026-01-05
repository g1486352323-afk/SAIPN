import pandas as pd
import os
import json
import networkx as nx
import datetime
import torch
import numpy as np
from tqdm import tqdm
import csv 
import argparse
import sys
import time

# ================= 1. 环境配置 =================
try:
    import cugraph
    import cudf
    HAS_GPU_GRAPH = True
    print("[Success] RAPIDS cuGraph Loaded.")
except ImportError:
    HAS_GPU_GRAPH = False
    print("[Info] cuGraph not found, using NetworkX (CPU).")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device: {device}")

# ================= 2. 路径配置 (已修正) =================
BASE_DIR = '/data_huawei/gaohaizhen/network/saipn/model/ablation-d2'
# [Output] 专门的消融实验输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'ablation_no_sentiment-D')

EMBEDDING_DIR = os.path.join(BASE_DIR, 'embedding')
NEW_VECTOR_PATH = os.path.join(EMBEDDING_DIR, 'output_vectors.txt')
NEW_SENTIMENT_PATH = os.path.join(EMBEDDING_DIR, 'final_with_sentiment.csv')
EMBEDDIN_ROOT = '/data_huawei/gaohaizhen/network/saipn/model'

if EMBEDDIN_ROOT not in sys.path:
    sys.path.append(EMBEDDIN_ROOT)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# 尝试导入 DCPRR
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
                # float16 省显存
                vec = torch.tensor([float(x) for x in parts[1].split()], dtype=torch.float16, device=device) 
                vector_dict[parts[0].strip()] = vec
            except: continue
    return vector_dict

def calculate_cnlr_fast(current_edges_set, prev_edges_set, communities_dict):
    """ 计算社区新链接率 (CNLR) """
    if not communities_dict or not current_edges_set: return 0.0
    new_edges = current_edges_set - prev_edges_set
    if not new_edges: return 0.0

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
    
    # --- A. 初始化输出 ---
    if os.path.exists(OUTPUT_DIR):
        for f in ["index_gpu.csv", "edge_log.csv"]:
            p = os.path.join(OUTPUT_DIR, f)
            if os.path.exists(p): os.remove(p)
    
    print(f"[Info] Output Directory: {OUTPUT_DIR}")

    # --- B. 加载数据 ---
    vec_path = args.vector_file if args.vector_file else NEW_VECTOR_PATH
    vector_dict = load_vectors_to_gpu(vec_path)
    
    # 虽然是消融实验，但情感值仍用于元数据存储(Metadata)，不参与向量拼接
    senti_path = args.sentiment_file if args.sentiment_file else NEW_SENTIMENT_PATH
    print(f"Loading sentiment: {senti_path}")
    senti_df = pd.read_csv(senti_path)
    
    id_col = next((c for c in senti_df.columns if 'id' in c and 'str' in c), 'id_str')
    senti_map = {}
    for _, row in senti_df.iterrows():
        pos, neu, neg = row['positive_probability'], row['neutral_probability'], row['negative_probability']
        val = 0.0
        if pos > max(neu, neg): val = pos
        elif neg > max(neu, pos): val = -neg
        else: val = (1-neu) if pos > neg else -(1-neu)
        senti_map[str(row[id_col])] = val

    data_file = args.data_file or NEW_SENTIMENT_PATH
    print(f"Loading data: {data_file}")
    table = pd.read_csv(data_file, dtype={'raw_value.id_str': str})
    table['raw_value.created_at'] = pd.to_datetime(table['raw_value.created_at'])
    table.sort_values('raw_value.created_at', inplace=True)
    table.set_index('raw_value.created_at', inplace=True, drop=False)

    # --- C. 全局变量 ---
    G = nx.DiGraph()
    MAX_NODES = 80000 
    
    # 获取 BERT 维度
    if vector_dict:
        bert_dim = list(vector_dict.values())[0].shape[0]
    else:
        bert_dim = 384
        
    # [关键修改] 维度 = BERT维度 (不加情感维)
    DIM = bert_dim 
    print(f"[Ablation Info] Vector Dimension: {DIM} (No Sentiment Dimension)")

    history_tensor = torch.zeros((MAX_NODES, DIM), dtype=torch.float16, device=device)
    history_times = torch.zeros(MAX_NODES, dtype=torch.float32, device=device)
    history_emotions = torch.zeros(MAX_NODES, dtype=torch.float16, device=device)
    history_ids = []  
    history_count = 0
    
    prev_edges_snapshot = set()
    all_stats_records = [] 

    csv_path = os.path.join(OUTPUT_DIR, "index_gpu.csv")
    f_csv = open(csv_path, 'w', newline='')
    writer = csv.writer(f_csv)
    headers = ['Time', 'Nodes', 'Edges', 'AvgPageRank', 'Assortativity', 'Modularity', 'DCPRR', 'CNLR', 'CompIntensity', 'PropScope', 'CollabIntensity']
    writer.writerow(headers)
    
    groups = list(table.resample(args.resample))
    print(f"[Info] Total Windows: {len(groups)}")

    # --- D. 循环 ---
    for w, w_t in tqdm(groups, desc="Simulating (No Sentiment)"):
        if len(w_t) == 0: continue
        
        current_ts = w.timestamp()
        cutoff_ts = current_ts - (args.delete_after_hours * 3600)
        
        # 1. 清理
        if G.number_of_nodes() > 0:
            nodes_to_remove = []
            for node, create_time in G.nodes(data='time'):
                if create_time is None: continue 
                if create_time < cutoff_ts:
                    nodes_to_remove.append(node)
            if nodes_to_remove:
                G.remove_nodes_from(nodes_to_remove)

        # 2. Batch 准备
        batch_ids = []
        batch_vecs = []
        batch_times = []
        batch_emos = [] 
        
        for _, row in w_t.iterrows():
            tid = str(row['raw_value.id_str'])
            ts = row['raw_value.created_at'].timestamp()
            emo = senti_map.get(tid, 0.0)
            
            if tid in vector_dict:
                b_vec = vector_dict[tid]
                
                # ================= [ABLATON KEY STEP] =================
                # 原始代码：comb = torch.cat((b_vec, e_tensor))
                # 消融代码：直接使用 b_vec，不拼接情感维度
                comb = b_vec 
                # ======================================================
                
                norm = torch.norm(comb)
                if norm > 0: comb /= norm
                
                batch_ids.append(tid)
                batch_vecs.append(comb)
                batch_times.append(ts)
                batch_emos.append(emo)
                
                G.add_node(tid, sentiment=float(emo), time=ts)
        
        if not batch_ids: continue

        batch_tensor = torch.stack(batch_vecs)
        batch_time_tensor = torch.tensor(batch_times, device=device)
        
        # 3. 计算 (逻辑不变，只是向量变短了)
        # History Interaction
        weights_hist = None
        h_mask = None
        if history_count > 0:
            curr_hist_tensor = history_tensor[:history_count]
            curr_hist_times = history_times[:history_count]
            
            sim_h = torch.mm(batch_tensor, curr_hist_tensor.t())
            diff_h = (batch_time_tensor.unsqueeze(1) - curr_hist_times.unsqueeze(0)) / 3600.0
            decay_h = torch.exp(-torch.abs(diff_h) / max(args.decay_unit_hours, 0.01))
            
            valid_time_mask = (curr_hist_times >= cutoff_ts).unsqueeze(0) 
            scores_h = sim_h * decay_h * valid_time_mask 
            h_mask = scores_h > args.score_threshold
            weights_hist = scores_h 

        # Intra-Batch Interaction
        sim_in = torch.mm(batch_tensor, batch_tensor.t())
        mask_in = torch.tril(torch.ones_like(sim_in), diagonal=-1).bool()
        diff_in = (batch_time_tensor.unsqueeze(1) - batch_time_tensor.unsqueeze(0)) / 3600.0
        decay_in = torch.exp(-torch.abs(diff_in) / max(args.decay_unit_hours, 0.01))
        scores_in = sim_in * decay_in * mask_in
        in_mask = scores_in > args.score_threshold

        # Propagation Loop
        batch_final_emos = []
        batch_final_vecs = []
        
        for i in range(len(batch_ids)):
            tid = batch_ids[i]
            current_emo = batch_emos[i]
            vol_update = 0.0
            
            # History
            if h_mask is not None:
                valid_h = h_mask[i] 
                if valid_h.any():
                    w_h = weights_hist[i][valid_h]
                    e_h = history_emotions[:history_count][valid_h]
                    contribs = w_h * e_h
                    vol_update += torch.mean(contribs).item()
                    
                    h_idx_list = torch.nonzero(valid_h).squeeze(1).cpu().tolist()
                    w_list = w_h.cpu().tolist()
                    for k, h_idx in enumerate(h_idx_list):
                        src_id = history_ids[h_idx]
                        if G.has_node(src_id):
                            G.add_edge(src_id, tid, weight=w_list[k])

            # Intra
            valid_in = in_mask[i]
            if valid_in.any():
                w_in = scores_in[i][valid_in]
                src_indices = torch.nonzero(valid_in).squeeze(1)
                src_updated_vals = [batch_final_emos[idx] for idx in src_indices.cpu().tolist()]
                src_updated_tensor = torch.tensor(src_updated_vals, device=device, dtype=torch.float16)
                
                contribs_in = w_in * src_updated_tensor
                vol_update += torch.mean(contribs_in).item()
                
                src_idx_list = src_indices.cpu().tolist()
                w_list_in = w_in.cpu().tolist()
                for k, src_idx in enumerate(src_idx_list):
                    src_id = batch_ids[src_idx]
                    G.add_edge(src_id, tid, weight=w_list_in[k])

            # Update Sentiment
            new_emo = current_emo + vol_update
            new_emo = max(-1.0, min(1.0, new_emo))
            batch_final_emos.append(new_emo)
            
            # Reconstruct Vector (ABLATION: Just BERT)
            # 同样，更新历史时也不拼接情感，只存 BERT
            new_vec = batch_tensor[i] 
            # 归一化已经在 batch 准备阶段做过了，这里其实不需要再做，但为了保险
            new_vec = new_vec / torch.norm(new_vec)
            
            batch_final_vecs.append(new_vec)
            G.nodes[tid]['sentiment'] = float(new_emo)

        # Write History
        if batch_final_vecs:
            n_new = len(batch_ids)
            if history_count + n_new < MAX_NODES:
                new_block_vec = torch.stack(batch_final_vecs)
                new_block_emo = torch.tensor(batch_final_emos, device=device, dtype=torch.float16)
                
                start = history_count
                end = history_count + n_new
                
                history_tensor[start:end] = new_block_vec
                history_times[start:end] = batch_time_tensor
                history_emotions[start:end] = new_block_emo 
                history_ids.extend(batch_ids)
                history_count += n_new

        # --- Metrics Calculation ---
        num_edges = G.number_of_edges()
        pr_val, mod_val, assort_val, dcprr_val, cnlr_val = 0, 0, 0, 0, 0
        comp_intensity, collab_intensity, prop_scope = 0, 0, 0
        communities_dict = {}

        # 1. GPU Graph
        USE_CUGRAPH = HAS_GPU_GRAPH and num_edges > 1000
        if USE_CUGRAPH:
            try:
                node_list = list(G.nodes())
                node_map = {n: i for i, n in enumerate(node_list)}
                df_edges = pd.DataFrame([(node_map[u], node_map[v], d['weight']) for u, v, d in G.edges(data=True)], columns=['src', 'dst', 'w'])
                gdf = cudf.DataFrame.from_pandas(df_edges)
                
                G_gpu = cugraph.Graph(directed=True)
                G_gpu.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w')
                pr_df = cugraph.pagerank(G_gpu)
                pr_val = pr_df['pagerank'].mean()

                G_gpu_und = cugraph.Graph(directed=False)
                G_gpu_und.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w')
                parts, mod_val = cugraph.louvain(G_gpu_und)
                
                parts_df = parts.to_pandas()
                inv_map = {v: k for k, v in node_map.items()}
                for row in parts_df.itertuples():
                    orig_id = inv_map.get(row.vertex)
                    if orig_id: communities_dict.setdefault(row.partition, []).append(orig_id)
            except: USE_CUGRAPH = False
        
        # 2. CPU Fallback
        if not USE_CUGRAPH and num_edges > 0:
            try:
                pr_val = float(np.mean(list(nx.pagerank(G).values())))
                import networkx.algorithms.community as nx_comm
                G_und = G.to_undirected()
                comm_sets = nx_comm.louvain_communities(G_und)
                mod_val = nx_comm.modularity(G_und, comm_sets)
                for i, c in enumerate(comm_sets): communities_dict[i] = list(c)
            except: pass

        # 3. Advanced Metrics
        if DCPRR_IMPORTED and communities_dict:
            try:
                scores = calculate_all_dcprr_scores(G, communities_dict)
                if scores: dcprr_val = float(np.mean(list(scores.values())))
            except: pass
        
        curr_edges_snapshot = set(G.edges())
        if prev_edges_snapshot:
            cnlr_val = calculate_cnlr_fast(curr_edges_snapshot, prev_edges_snapshot, communities_dict)
        prev_edges_snapshot = curr_edges_snapshot

        try: 
            assort_val = nx.degree_assortativity_coefficient(G)
            if np.isnan(assort_val): assort_val = 0.0 
        except: assort_val = 0.0
        
        try: comp_intensity = sum(d['weight'] for u,v,d in G.edges(data=True))
        except: comp_intensity = 0
        try: 
            if num_edges < 500000: collab_intensity = int(sum(nx.triangles(G.to_undirected()).values())/3)
        except: collab_intensity = 0
        try:
             prop_scope = max((len(c) for c in (nx.weakly_connected_components(G) if nx.is_directed(G) else nx.connected_components(G))), default=0)
        except: prop_scope = 0

        row_data = [
            w.strftime('%Y-%m-%d'),
            G.number_of_nodes(), num_edges,
            f"{pr_val:.4f}", f"{assort_val:.4f}", f"{mod_val:.4f}",
            f"{dcprr_val:.4f}", f"{cnlr_val:.4f}",
            f"{comp_intensity:.2f}", prop_scope, collab_intensity
        ]
        writer.writerow(row_data)
        f_csv.flush() 
        all_stats_records.append([float(x) for x in row_data[1:]])

    # --- E. 汇总 ---
    if all_stats_records:
        data_matrix = np.array(all_stats_records)
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)
        nodes_col = data_matrix[:, 0]
        edges_col = data_matrix[:, 1]
        
        final_vals = []
        # 0:Nodes, 1:Edges, 2:AvgPageRank, 3:Assortativity, 4:Modularity, 
        # 5:DCPRR, 6:CNLR, 7:CompIntensity, 8:PropScope, 9:CollabIntensity
        for col_idx in range(data_matrix.shape[1]):
            col_data = data_matrix[:, col_idx]
            val = 0.0
            if col_idx in [0, 1, 7, 9]: # 绝对量 -> 平均
                val = np.mean(col_data)
            elif col_idx in [2, 8]: # 节点属性 -> 节点加权
                val = np.average(col_data, weights=nodes_col) if np.sum(nodes_col) > 0 else 0.0
            elif col_idx in [3, 4, 5]: # 结构属性 -> 边加权
                val = np.average(col_data, weights=edges_col) if np.sum(edges_col) > 0 else 0.0
            elif col_idx == 6: # CNLR -> Non-zero mean
                valid = col_data > 1e-6
                val = np.mean(col_data[valid]) if np.any(valid) else 0.0
            final_vals.append(val)

        avg_row = ['GLOBAL_AVERAGE_WEIGHTED'] + [f"{v:.4f}" for v in final_vals]
        writer.writerow(avg_row)
        print("\n[Success] Ablation study completed. Weighted averages saved.")
    else:
        print("\n[Warning] No data processed.")

    f_csv.close()
    print(f"Results saved to {OUTPUT_DIR}")
    print(f"Total time: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resample', default='D') 
    parser.add_argument('--decay-unit-hours', type=float, default=48.0)
    parser.add_argument('--delete-after-hours', type=float, default=72.0)
    parser.add_argument('--score-threshold', type=float, default=0.70)
    parser.add_argument('--data-file', default=None)
    parser.add_argument('--vector-file', default=None)
    parser.add_argument('--sentiment-file', default=None)
    parser.add_argument('--save-matrix', action='store_true') 
    
    args = parser.parse_args()
    main_gpu(args)