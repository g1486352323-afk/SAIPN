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
import shutil

# ================= 1. 环境与路径配置 =================
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

# 路径配置
BASE_DIR = '/data_huawei/gaohaizhen/network/saipn/model/ablation-d2'
EMBEDDING_DIR = os.path.join(BASE_DIR, 'embedding')
NEW_VECTOR_PATH = os.path.join(EMBEDDING_DIR, 'output_vectors.txt')
NEW_SENTIMENT_PATH = os.path.join(EMBEDDING_DIR, 'final_with_sentiment2.csv')
EMBEDDIN_ROOT = '/data_huawei/gaohaizhen/network/saipn/model'

if EMBEDDIN_ROOT not in sys.path:
    sys.path.append(EMBEDDIN_ROOT)

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
                # 预先转为 Tensor 存储 (float16 省显存)
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
        
        # 统计新增边中有多少条在社区内部
        internal_new = sum(1 for u, v in new_edges if u in node_set and v in node_set)
        cnlr_scores.append(internal_new / possible)
    
    val = float(np.mean(cnlr_scores)) if cnlr_scores else 0.0
    return 0.0 if np.isnan(val) else val

# ================= 3. 主逻辑 =================

def main_gpu(args):
    start_time = time.time()
    
    # --- A. 初始化输出目录 ---
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
    else:
        OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'implicit-ablation-D')
    
    if os.path.exists(OUTPUT_DIR):
        for f in ["index_gpu.csv", "edge_log.csv"]:
            p = os.path.join(OUTPUT_DIR, f)
            if os.path.exists(p): os.remove(p)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 新增：为 CNLR 保存隐式网络快照的目录
    SNAPSHOT_DIR = os.path.join(OUTPUT_DIR, 'snapshots')
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    # --- B. 加载数据 ---
    vec_path = args.vector_file if args.vector_file else NEW_VECTOR_PATH
    vector_dict = load_vectors_to_gpu(vec_path)
    
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

    data_file = args.data_file or os.path.join(BASE_DIR, "/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/embedding/final_with_sentiment2.csv")
    print(f"Loading data: {data_file}")
    table = pd.read_csv(data_file, dtype={'raw_value.id_str': str})
    table['raw_value.created_at'] = pd.to_datetime(table['raw_value.created_at'])
    
    # [关键] 按时间排序，设置索引并保留列
    table.sort_values('raw_value.created_at', inplace=True)
    table.set_index('raw_value.created_at', inplace=True, drop=False)

    # --- C. 全局状态变量 ---
    G = nx.DiGraph()
    MAX_NODES = 80000 
    DIM = 385           
    
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
    # Header 定义 (共11列)
    headers = ['Time', 'Nodes', 'Edges', 'AvgPageRank', 'Assortativity', 'Modularity', 'DCPRR', 'CNLR', 'CompIntensity', 'PropScope', 'CollabIntensity']
    writer.writerow(headers)
    
    groups = list(table.resample(args.resample))
    print(f"[Info] Total Windows: {len(groups)}")

    # --- D. 时间步循环 ---
    step_idx = 0
    last_metrics = {
        'pr_val': 0.0,
        'mod_val': 0.0,
        'assort_val': 0.0,
        'dcprr_val': 0.0,
        'cnlr_val': 0.0,
        'comp_intensity': 0.0,
        'collab_intensity': 0.0,
        'prop_scope': 0,
    }

    for w, w_t in tqdm(groups, desc="Simulating"):
        if len(w_t) == 0:
            continue

        step_idx += 1
        compute_metrics_now = (step_idx == 1) or (step_idx % args.metric_interval == 0)
        
        current_ts = w.timestamp()
        cutoff_ts = current_ts - (args.delete_after_hours * 3600)
        
        # 1. 严格图清理
        if G.number_of_nodes() > 0:
            nodes_to_remove = []
            for node, create_time in G.nodes(data='time'):
                if create_time is None: continue 
                if create_time < cutoff_ts:
                    nodes_to_remove.append(node)
            if nodes_to_remove:
                G.remove_nodes_from(nodes_to_remove)

        # 2. 准备 Batch
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
        
        if not batch_ids: continue

        batch_tensor = torch.stack(batch_vecs)
        batch_time_tensor = torch.tensor(batch_times, device=device)
        
        # 3. 混合模式计算
        # Step 1: 拓扑 (GPU)
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

        # Step 2: 传播 (CPU Loop)
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

            # Intra-Batch
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

            # Update
            new_emo = current_emo + vol_update
            new_emo = max(-1.0, min(1.0, new_emo))
            batch_final_emos.append(new_emo)
            
            # Reconstruct
            bert_part = batch_tensor[i][:-1] 
            new_emo_t = torch.tensor([new_emo], device=device, dtype=torch.float16)
            new_vec = torch.cat((bert_part, new_emo_t))
            new_vec = new_vec / torch.norm(new_vec)
            batch_final_vecs.append(new_vec)
            G.nodes[tid]['sentiment'] = float(new_emo)

        # Step 3: Write History
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

        # ================= [指标计算] =================
        num_edges = G.number_of_edges()
        
        pr_val, mod_val, assort_val, dcprr_val, cnlr_val = 0, 0, 0, 0, 0
        comp_intensity, collab_intensity, prop_scope = 0, 0, 0

        if compute_metrics_now and num_edges > 0:
            communities_dict = {}

            # 1. GPU Calc
            USE_CUGRAPH = HAS_GPU_GRAPH and num_edges > 1000
            if USE_CUGRAPH:
                try:
                    node_list = list(G.nodes())
                    node_map = {n: i for i, n in enumerate(node_list)}
                    df_edges = pd.DataFrame([(node_map[u], node_map[v], d['weight']) for u, v, d in G.edges(data=True)], columns=['src', 'dst', 'w'])
                    
                    gdf = cudf.DataFrame.from_pandas(df_edges)
                    G_gpu = cugraph.Graph(directed=True)
                    try:
                        G_gpu.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w', store_transposed=True)
                    except TypeError:
                        # 兼容旧版本 cuGraph 不支持 store_transposed 参数的情况
                        G_gpu.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w')

                    pr_df = cugraph.pagerank(G_gpu)
                    pr_val = pr_df['pagerank'].mean()

                    G_gpu_und = cugraph.Graph(directed=False)
                    try:
                        G_gpu_und.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w', store_transposed=True)
                    except TypeError:
                        G_gpu_und.from_cudf_edgelist(gdf, source='src', destination='dst', edge_attr='w')
                    parts, mod_val = cugraph.louvain(G_gpu_und)
                    
                    parts_df = parts.to_pandas()
                    inv_map = {v: k for k, v in node_map.items()}
                    for row in parts_df.itertuples():
                        orig_id = inv_map.get(row.vertex)
                        if orig_id: communities_dict.setdefault(row.partition, []).append(orig_id)
                except Exception as e:
                    USE_CUGRAPH = False
            
            # 2. CPU Fallback
            if not USE_CUGRAPH and num_edges > 0:
                try:
                    pr_val = float(np.mean(list(nx.pagerank(G).values())))
                    import networkx.algorithms.community as nx_comm
                    G_und = G.to_undirected()
                    comm_sets = nx_comm.louvain_communities(G_und)
                    mod_val = nx_comm.modularity(G_und, comm_sets)
                    for i, c in enumerate(comm_sets):
                        communities_dict[i] = list(c)
                except: pass

            # 3. Metrics
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
                if num_edges < 500000: 
                    collab_intensity = int(sum(nx.triangles(G.to_undirected()).values())/3)
            except: collab_intensity = 0
            try:
                 prop_scope = max((len(c) for c in (nx.weakly_connected_components(G) if nx.is_directed(G) else nx.connected_components(G))), default=0)
            except: prop_scope = 0

            # 4. 记录最近一次完整指标，用于中间步复用
            last_metrics = {
                'pr_val': pr_val,
                'mod_val': mod_val,
                'assort_val': assort_val,
                'dcprr_val': dcprr_val,
                'cnlr_val': cnlr_val,
                'comp_intensity': comp_intensity,
                'collab_intensity': collab_intensity,
                'prop_scope': prop_scope,
            }
        else:
            # 复用上一次的指标值，仅更新节点/边规模信息
            pr_val           = last_metrics['pr_val']
            mod_val          = last_metrics['mod_val']
            assort_val       = last_metrics['assort_val']
            dcprr_val        = last_metrics['dcprr_val']
            cnlr_val         = last_metrics['cnlr_val']
            comp_intensity   = last_metrics['comp_intensity']
            collab_intensity = last_metrics['collab_intensity']
            prop_scope       = last_metrics['prop_scope']

        # 4. Write CSV (Every Step)
        row_data = [
            w.strftime('%Y-%m-%d'),
            G.number_of_nodes(), num_edges,
            f"{pr_val:.4f}", f"{assort_val:.4f}", f"{mod_val:.4f}",
            f"{dcprr_val:.4f}", f"{cnlr_val:.4f}",
            f"{comp_intensity:.2f}", prop_scope, collab_intensity
        ]
        writer.writerow(row_data)
        f_csv.flush() 

        # 4.1 保存当前图的边列表快照（供 CNLR 使用）
        if num_edges > 0:
            # 使用日期+时间的时间戳格式，方便 count_CNLR.parse_timestamp 解析
            ts_str = w.strftime('%Y-%m-%d_%H-%M')
            filename = f"implicit_edges_{ts_str}.edgelist"
            filepath = os.path.join(SNAPSHOT_DIR, filename)
            try:
                # 只保存权重属性，减小体积
                nx.write_edgelist(G, filepath, data=['weight'])
            except Exception as e:
                print(f"[Warning] Failed to save snapshot {filename}: {e}")

        # Record for Global Calculation (Skip Time column)
        vals = [float(x) for x in row_data[1:]]
        all_stats_records.append(vals)

    # --- E. End: Global Average (Scientific Weighted Calculation) ---
    if all_stats_records:
        # [严谨性修正] 使用混合加权策略，避免长尾效应偏差
        data_matrix = np.array(all_stats_records)
        data_matrix = np.nan_to_num(data_matrix, nan=0.0)

        # 列索引映射 (基于 row_data[1:]):
        # 0:Nodes, 1:Edges, 2:AvgPageRank, 3:Assortativity, 4:Modularity, 
        # 5:DCPRR, 6:CNLR, 7:CompIntensity, 8:PropScope, 9:CollabIntensity
        
        nodes_col = data_matrix[:, 0]
        edges_col = data_matrix[:, 1]
        
        total_nodes = np.sum(nodes_col)
        total_edges = np.sum(edges_col)
        
        final_vals = []
        
        for col_idx in range(data_matrix.shape[1]):
            col_data = data_matrix[:, col_idx]
            val = 0.0
            
            # 策略 1: 绝对规模类 (Nodes, Edges, CompIntensity, CollabIntensity) -> 简单平均
            # 理由: 反映日均网络规模
            if col_idx in [0, 1, 7, 9]: 
                val = np.mean(col_data)
                
            # 策略 2: 节点属性类 (AvgPageRank, PropScope) -> 节点数加权
            # 理由: 避免微型网络的极端值干扰全局平均
            elif col_idx in [2, 8]:
                val = np.average(col_data, weights=nodes_col) if total_nodes > 0 else 0.0

            # 策略 3: 结构质量类 (Assortativity, Modularity, DCPRR) -> 边数加权
            # 理由: 结构依附于连边，大网络的结构质量权重应更高
            elif col_idx in [3, 4, 5]:
                val = np.average(col_data, weights=edges_col) if total_edges > 0 else 0.0
            
            # 策略 4: 演化率类 (CNLR) -> 非零平均 (Non-zero Mean)
            # 理由: CNLR=0 代表无演化事件(静默期)，属于无效样本，应剔除
            elif col_idx == 6:
                valid_mask = col_data > 1e-6
                if np.any(valid_mask):
                    val = np.mean(col_data[valid_mask])
                else:
                    val = 0.0
            
            final_vals.append(val)

        # 写入最终行
        avg_row = ['GLOBAL_AVERAGE_WEIGHTED'] + [f"{v:.4f}" for v in final_vals]
        writer.writerow(avg_row)
        print("\n[Success] Global averages written to CSV (Weighted Strategy applied).")
    else:
        print("\n[Warning] No data processed.")

    f_csv.close()
    print(f"Done. Results saved to {OUTPUT_DIR}")
    print(f"Total execution time: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resample', default='D') 
    parser.add_argument('--decay-unit-hours', type=float, default=360.0)
    parser.add_argument('--delete-after-hours', type=float, default=720.0)
    parser.add_argument('--score-threshold', type=float, default=0.70)
    parser.add_argument('--data-file', default=None)
    parser.add_argument('--vector-file', default=None)
    parser.add_argument('--sentiment-file', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--save-matrix', action='store_true') 
    parser.add_argument('--metric-interval', type=int, default=1, help='Compute heavy graph metrics every N windows (default=1)')
    
    args = parser.parse_args()
    main_gpu(args)