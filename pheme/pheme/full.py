import pandas as pd
import os
import networkx as nx
import torch
import numpy as np
from tqdm import tqdm
import csv 
import argparse
import sys
import time
import warnings
import shutil

# ================= 1. 环境与配置 =================
warnings.filterwarnings("ignore")

# 混合模式：GPU 算向量，CPU 算图
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Compute Device (Vectors): {device}")
print(f"Graph Compute: NetworkX (CPU) - High Precision Mode")

# 路径配置 (请根据实际情况保留或修改默认值)
BASE_DIR = '/data_huawei/gaohaizhen/network/saipn/model/ablation'
NEW_VECTOR_PATH = os.path.join(BASE_DIR, 'embeddin', 'output_vectors.txt')
NEW_SENTIMENT_PATH = os.path.join(BASE_DIR, 'embeddin', 'final_with_sentiment.csv')
EMBEDDIN_ROOT = os.path.join(BASE_DIR, 'embeddin')

if EMBEDDIN_ROOT not in sys.path: sys.path.append(EMBEDDIN_ROOT)

# 尝试导入 DCPRR
try:
    from baseline_propagation.metrics import calculate_all_dcprr_scores
    DCPRR_IMPORTED = True
except ImportError:
    DCPRR_IMPORTED = False
    def calculate_all_dcprr_scores(*args, **kwargs): return {}

# ================= 2. 辅助函数 =================

def load_vectors_to_gpu(file_path):
    if not os.path.exists(file_path): return {}
    vector_dict = {}
    print(f"Loading vectors: {file_path} ...")
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            try:
                # 向量放 GPU，加速矩阵乘法
                vec = torch.tensor([float(x) for x in parts[1].split()], dtype=torch.float16, device=device) 
                vector_dict[parts[0].strip()] = vec
            except: continue
    return vector_dict

def calculate_cnlr_fast(current_edges_set, prev_edges_set, communities_dict):
    """ 计算社区新链接率 (CNLR) """
    if not communities_dict or not current_edges_set: return 0.0
    new_edges = current_edges_set - prev_edges_set
    if not new_edges: return 0.0
    
    # 性能保护
    if len(new_edges) > 100000: return 0.0

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

# ================= 3. 主逻辑 (CPU 优化版) =================

def main_optimized(args):
    start_time = time.time()
    
    # --- A. 初始化输出目录与覆盖逻辑 ---
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'implicit-ablation-HH')
    
    # [新增] Force 参数逻辑：存在则强制删除重建
    if os.path.exists(OUTPUT_DIR):
        if args.force:
            print(f"[Info] --force detected. Removing old directory: {OUTPUT_DIR}")
            try:
                shutil.rmtree(OUTPUT_DIR)
            except Exception as e:
                print(f"[Warning] Failed to remove dir: {e}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        else:
            # 如果没有 force 且文件已存在，跳过
            if os.path.exists(os.path.join(OUTPUT_DIR, "index_gpu.csv")):
                print(f"[Info] Result already exists. Skipping... (Use --force to overwrite)")
                return
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # [新增] 创建快照目录用于保存边列表
    SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, 'snapshots')
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # --- B. 加载数据 ---
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

    data_file = args.data_file or os.path.join(BASE_DIR, "charliehebdo_gemini_2_flash_output_fixed_from_cleaned.csv")
    print(f"Loading data: {data_file}")
    table = pd.read_csv(data_file, dtype={'raw_value.id_str': str})
    table['raw_value.created_at'] = pd.to_datetime(table['raw_value.created_at'])
    
    table.sort_values('raw_value.created_at', inplace=True)
    table.set_index('raw_value.created_at', inplace=True, drop=False)

    # --- C. 全局状态变量 ---
    G = nx.DiGraph()
    MAX_NODES = 80000 
    
    # 历史记录用 GPU Tensor 存储
    history_tensor = torch.zeros((MAX_NODES, 385), dtype=torch.float16, device=device)
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
    
    # [关键] 强制使用小时级(H)进行演化
    resample_granularity = args.resample  # 默认为 'H'
    groups = list(table.resample(resample_granularity))
    
    print(f"[Info] Total Evolution Steps: {len(groups)} (Granularity: {resample_granularity})")
    print(f"[Info] Observation Strategy: Snapshot every 6 hours")

    # --- D. 时间步循环 (每小时演化) ---
    for loop_idx, (w, w_t) in enumerate(tqdm(groups, desc="Simulating")):
        
        # 1. 严格图清理 (CPU) - 即使没数据也要清理过期节点
        current_ts = w.timestamp()
        cutoff_ts = current_ts - (args.delete_after_hours * 3600)
        
        if G.number_of_nodes() > 0:
            nodes_to_remove = []
            for node, create_time in G.nodes(data='time'):
                if create_time is None: continue 
                if create_time < cutoff_ts:
                    nodes_to_remove.append(node)
            if nodes_to_remove:
                G.remove_nodes_from(nodes_to_remove)

        # 2. 图的生长 (Graph Growth) - 只要有数据就更新
        if len(w_t) > 0:
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
                    e_tensor = torch.tensor([emo], device=device, dtype=torch.float16)
                    comb = torch.cat((b_vec, e_tensor))
                    norm = torch.norm(comb)
                    if norm > 0: comb /= norm
                    
                    batch_ids.append(tid)
                    batch_vecs.append(comb)
                    batch_times.append(ts)
                    batch_emos.append(emo)
                    
                    G.add_node(tid, sentiment=float(emo), time=ts)
            
            if batch_ids:
                batch_tensor = torch.stack(batch_vecs)
                batch_time_tensor = torch.tensor(batch_times, device=device)
                
                # --- GPU 矩阵计算与连边逻辑 ---
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

                sim_in = torch.mm(batch_tensor, batch_tensor.t())
                mask_in = torch.tril(torch.ones_like(sim_in), diagonal=-1).bool()
                diff_in = (batch_time_tensor.unsqueeze(1) - batch_time_tensor.unsqueeze(0)) / 3600.0
                decay_in = torch.exp(-torch.abs(diff_in) / max(args.decay_unit_hours, 0.01))
                scores_in = sim_in * decay_in * mask_in
                in_mask = scores_in > args.score_threshold

                # --- 传播与连边 (CPU Loop) ---
                batch_final_emos = []
                batch_final_vecs = []
                
                for i in range(len(batch_ids)):
                    tid = batch_ids[i]
                    current_emo = batch_emos[i]
                    vol_update = 0.0
                    
                    # History Edges
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
                                    G.add_edge(src_id, tid, weight=float(w_list[k]))

                    # Intra-Batch Edges
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
                            G.add_edge(src_id, tid, weight=float(w_list_in[k]))

                    new_emo = max(-1.0, min(1.0, current_emo + vol_update))
                    batch_final_emos.append(new_emo)
                    
                    bert_part = batch_tensor[i][:-1] 
                    new_emo_t = torch.tensor([new_emo], device=device, dtype=torch.float16)
                    new_vec = torch.cat((bert_part, new_emo_t))
                    new_vec = new_vec / torch.norm(new_vec)
                    batch_final_vecs.append(new_vec)
                    G.nodes[tid]['sentiment'] = float(new_emo)

                # Update History Tensor
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

        # ================= [关键修改] 降频观测: 每 6 小时计算一次 =================
        # loop_idx 0 (00:00), 6 (06:00), 12 (12:00)...
        if loop_idx % 6 == 0:
            num_edges = G.number_of_edges()
            
            pr_val, mod_val, assort_val, dcprr_val, cnlr_val = 0, 0, 0, 0, 0
            comp_intensity, collab_intensity, prop_scope = 0, 0, 0
            communities_dict = {}

            # 1. PageRank & Community
            if num_edges > 0:
                try:
                    pr_val = float(np.mean(list(nx.pagerank(G).values())))
                except: pass
                
                try:
                    import networkx.algorithms.community as nx_comm
                    G_und = G.to_undirected()
                    comm_sets = nx_comm.louvain_communities(G_und)
                    mod_val = nx_comm.modularity(G_und, comm_sets)
                    for i, c in enumerate(comm_sets):
                        communities_dict[i] = list(c)
                except: pass

            # 2. Advanced Metrics (DCPRR)
            if DCPRR_IMPORTED and communities_dict:
                try:
                    scores = calculate_all_dcprr_scores(G, communities_dict)
                    if scores: dcprr_val = float(np.mean(list(scores.values())))
                except: pass
            
            # 3. CNLR (Change Rate since LAST SNAPSHOT, i.e., 6 hours ago)
            curr_edges_snapshot = set(G.edges())
            if prev_edges_snapshot:
                cnlr_val = calculate_cnlr_fast(curr_edges_snapshot, prev_edges_snapshot, communities_dict)
            # 更新快照
            prev_edges_snapshot = curr_edges_snapshot

            # 4. Structural Metrics
            try: 
                if num_edges < 50000:
                    assort_val = nx.degree_assortativity_coefficient(G)
                    if np.isnan(assort_val): assort_val = 0.0 
            except: assort_val = 0.0
            
            try: comp_intensity = G.size(weight='weight')
            except: comp_intensity = 0
            
            try: 
                if num_edges < 30000: 
                    collab_intensity = int(sum(nx.triangles(G.to_undirected()).values())/3)
            except: collab_intensity = 0
            
            try:
                 prop_scope = max((len(c) for c in (nx.weakly_connected_components(G) if nx.is_directed(G) else nx.connected_components(G))), default=0)
            except: prop_scope = 0

            # 5. Write CSV (Only on sampled hours)
            row_data = [
                w.strftime('%Y-%m-%d %H:%M:%S'),
                G.number_of_nodes(), num_edges,
                f"{pr_val:.4f}", f"{assort_val:.4f}", f"{mod_val:.4f}",
                f"{dcprr_val:.4f}", f"{cnlr_val:.4f}",
                f"{comp_intensity:.2f}", prop_scope, collab_intensity
            ]
            writer.writerow(row_data)
            f_csv.flush() 

            # [新增] 保存当前图的边列表快照（仅在图非空时）
            if num_edges > 0:
                ts_str = w.strftime('%Y-%m-%d_%H-%M')
                filename = f"implicit_edges_{ts_str}.edgelist"
                filepath = os.path.join(SNAPSHOT_DIR, filename)
                try:
                    # 只保存权重属性，减小体积
                    nx.write_edgelist(G, filepath, data=['weight'])
                except Exception as e:
                    print(f"[Warning] Failed to save snapshot {filename}: {e}")

            # Record for Global Average
            vals = [float(x) for x in row_data[1:]]
            all_stats_records.append(vals)

    # --- E. End: Global Average ---
    if all_stats_records:
        data_matrix = np.array(all_stats_records)
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)

        nodes_col = data_matrix[:, 0]
        edges_col = data_matrix[:, 1]
        
        total_nodes = np.sum(nodes_col)
        total_edges = np.sum(edges_col)
        
        final_vals = []
        
        for col_idx in range(data_matrix.shape[1]):
            col_data = data_matrix[:, col_idx]
            val = 0.0
            if col_idx in [0, 1, 7, 9]: 
                val = np.mean(col_data)
            elif col_idx in [2, 8]:
                val = np.average(col_data, weights=nodes_col) if total_nodes > 0 else 0.0
            elif col_idx in [3, 4, 5]:
                val = np.average(col_data, weights=edges_col) if total_edges > 0 else 0.0
            elif col_idx == 6: # DCPRR
                valid_mask = col_data > 1e-6
                val = np.mean(col_data[valid_mask]) if np.any(valid_mask) else 0.0
            final_vals.append(val)

        avg_row = ['GLOBAL_AVERAGE_WEIGHTED'] + [f"{v:.4f}" for v in final_vals]
        writer.writerow(avg_row)
        print("\n[Success] Global averages written to CSV.")
    else:
        print("\n[Warning] No data processed.")

    f_csv.close()
    print(f"Done. Results saved to {OUTPUT_DIR}")
    print(f"Total execution time: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resample', default='H', help="Evolution granularity (always keep H)") 
    parser.add_argument('--decay-unit-hours', type=float, default=48.0)
    parser.add_argument('--delete-after-hours', type=float, default=72.0)
    parser.add_argument('--score-threshold', type=float, default=0.70)
    parser.add_argument('--data-file', default=None)
    parser.add_argument('--vector-file', default=None)
    parser.add_argument('--sentiment-file', default=None)
    parser.add_argument('--output-dir', default=None)
    
    # [新增] 强制覆盖开关
    parser.add_argument('--force', action='store_true', help="Force overwrite output directory if exists")
    
    args = parser.parse_args()
    main_optimized(args)