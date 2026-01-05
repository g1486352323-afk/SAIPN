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

# ================= 1. GPU 环境检查与导入 =================
try:
    import cugraph
    import cudf
    print("[Success] 成功导入 RAPIDS cuGraph (GPU 图计算库)")
    HAS_GPU_GRAPH = True
except ImportError:
    print("[Error] 未检测到 cugraph，将使用 NetworkX (CPU)")
    HAS_GPU_GRAPH = False

# 检查 PyTorch GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用计算设备: {device}")

# ================= 2. 配置路径 =================
BASE_DIR = '/data_huawei/gaohaizhen/network/saipn/model/ablation'

# [关键修改 1] 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'ablation_no_tags-H')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [关键修改 2] 使用 No-Tags 向量文件
NEW_VECTOR_PATH = os.path.join(BASE_DIR, 'embeddin', 'output_vectors_no_tags.txt')
NEW_SENTIMENT_PATH = os.path.join(BASE_DIR, 'embeddin', 'final_with_sentiment.csv')

EMBEDDIN_ROOT = '/data_huawei/gaohaizhen/network/saipn/model'
if EMBEDDIN_ROOT not in sys.path:
    sys.path.append(EMBEDDIN_ROOT)

# 尝试导入 DCPRR 计算函数
DCPRR_IMPORTED = False
try:
    from baseline_propagation.metrics import calculate_all_dcprr_scores
    DCPRR_IMPORTED = True
except ImportError:
    print("[Warning] 无法导入 baseline_propagation，DCPRR 将为 0")
    def calculate_all_dcprr_scores(*args, **kwargs): return {}

# ================= 3. 核心辅助函数 =================

def load_vectors_to_gpu(file_path):
    vector_dict = {}
    print(f"正在加载向量文件: {file_path} ...")
    if not os.path.exists(file_path):
        print(f"[Error] 找不到向量文件: {file_path}")
        return {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',', 1)
            if len(parts) < 2: continue
            tweet_id = parts[0].strip()
            try:
                vec_data = [float(x) for x in parts[1].split()]
                vector = torch.tensor(vec_data, dtype=torch.float32, device=device)
                vector_dict[tweet_id] = vector
            except ValueError:
                continue
    return vector_dict

def get_combined_embedding(tweet_id, emotion_score, vector_dict):
    """
    [Tag消融]：向量已经是无Tag的语义向量，这里依然拼接情感
    """
    tweet_id = str(tweet_id)
    if tweet_id not in vector_dict:
        return None
    
    bert_vec = vector_dict[tweet_id]
    emo_tensor = torch.tensor([emotion_score], dtype=torch.float32, device=device)
    
    # 正常拼接：(无Tag语义) + 情感
    combined = torch.cat((bert_vec, emo_tensor), dim=0)
    
    norm = torch.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined

def calculate_vol_update(parent_sims, parent_times, parent_emotions, current_time_ts, current_emotion, decay_unit):
    if len(parent_sims) == 0:
        return current_emotion
    time_diffs = (current_time_ts - parent_times) / 3600.0
    tau = max(decay_unit, 0.01)
    daw_values = torch.exp(-torch.abs(time_diffs) / tau)
    daw_values = torch.clamp(daw_values, min=0.01)
    contributions = daw_values * parent_sims * parent_emotions
    vol1 = torch.mean(contributions).item()
    new_emotion = current_emotion + vol1
    if new_emotion < -1: return -1.0
    if new_emotion > 1: return 1.0
    return new_emotion

# [新增] CNLR 计算函数
def calculate_cnlr(current_edges_set, prev_edges_set, communities_dict):
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

# ================= 4. 主逻辑 =================

def main_gpu(args):
    # [修复] 运行前先清理旧的 CSV 文件，防止列数冲突报错
    index_path = os.path.join(OUTPUT_DIR, "index_gpu.csv")
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
            print(f"[Info] 已清理旧统计文件: {index_path}")
        except: pass

    start_time_global = time.time()
    
    # --- A. 数据加载 ---
    vec_path = args.vector_file if args.vector_file else NEW_VECTOR_PATH
    vector_dict = load_vectors_to_gpu(vec_path)
    if not vector_dict:
        print("程序终止：无法加载向量。")
        return

    print(f"[Info] 已加载 {len(vector_dict)} 条向量到显存")
    
    if len(vector_dict) > 0:
        sample_dim = next(iter(vector_dict.values())).shape[0] + 1
    else:
        sample_dim = 385 
    print(f"[Info] 向量维度 (含情感): {sample_dim}")

    senti_path = args.sentiment_file if args.sentiment_file else NEW_SENTIMENT_PATH
    print(f"正在加载情感数据: {senti_path}")
    if not os.path.exists(senti_path):
        print(f"[Error] 找不到情感文件: {senti_path}")
        return

    sentiment_df = pd.read_csv(senti_path, dtype={'raw_value.id_str': str})
    
    senti_id_col = 'raw_value.id_str' if 'raw_value.id_str' in sentiment_df.columns else 'id_str'
    if senti_id_col not in sentiment_df.columns:
         for col in sentiment_df.columns:
             if 'id' in col and 'str' in col:
                 senti_id_col = col
                 break
    
    sentiment_dict = {
        str(row[senti_id_col]): {
            'pos': row['positive_probability'],
            'neu': row['neutral_probability'],
            'neg': row['negative_probability']
        }
        for _, row in sentiment_df.iterrows()
    }

    def get_raw_emotion(tid):
        d = sentiment_dict.get(str(tid))
        if not d: return 0.0
        if d['pos'] > d['neu'] and d['pos'] > d['neg']: return d['pos']
        if d['neg'] > d['neu'] and d['neg'] > d['pos']: return -d['neg']
        if d['neu'] > d['pos'] and d['neu'] > d['neg']:
            return (1 - d['neu']) if d['pos'] > d['neg'] else -(1 - d['neu'])
        return 0.0

    data_file = args.data_file or os.path.join(BASE_DIR, "charliehebdo_gemini_2_flash_output_fixed_from_cleaned.csv")
    print(f"正在读取主数据文件: {data_file}")
    if not os.path.exists(data_file):
        print(f"[Error] 主数据文件不存在: {data_file}")
        return

    table = pd.read_csv(data_file, dtype={'raw_value.id_str': str, 'raw_value.user_id_str': str})
    if table.columns[0].lower().startswith('unnamed'):
        table = table.iloc[:, 1:]
    
    table['raw_value.created_at'] = pd.to_datetime(table['raw_value.created_at'])
    table = table.sort_values(by='raw_value.created_at')
    table.index = table['raw_value.created_at']

    # --- B. 初始化变量 ---
    G = nx.DiGraph() 
    active_history = [] 
    users_emotion_history = {}
    tweet_to_user = {} 
    first_appearance = {}
    total_edges_all_time = 0
    
    # [关键修复] 初始化快照变量，用于 CNLR 计算
    prev_edges_snapshot = set()
    
    edge_log_path = os.path.join(OUTPUT_DIR, "edge_log.csv")
    edge_log_file = open(edge_log_path, "w", newline='', encoding="utf-8")
    edge_writer = csv.writer(edge_log_file)
    edge_writer.writerow(['Source', 'Target', 'Time', 'Weight'])

    stats_buffer = []
    groups = list(table.resample(args.resample))
    print(f"[Info] 开始处理，总窗口数: {len(groups)}，频率: {args.resample}")

    # --- C. 循环 ---
    for w, w_t in tqdm(groups, desc="Progress", unit="win"):
        if len(w_t) == 0: continue
        
        current_window_ts = w.timestamp()
        cutoff_ts = current_window_ts - (args.delete_after_hours * 3600)

        # 1. 维护滑动窗口
        active_history = [item for item in active_history if item['time'] >= cutoff_ts]
        valid_ids = set(item['id'] for item in active_history)
        current_batch_ids = set(w_t['raw_value.id_str'].astype(str).values)
        
        nodes_to_remove = [n for n in G.nodes if n not in valid_ids and n not in current_batch_ids]
        if nodes_to_remove:
            G.remove_nodes_from(nodes_to_remove)

        # ----------------- 预分配显存 -----------------
        hist_len = len(active_history)
        curr_len = len(w_t)
        total_capacity = hist_len + curr_len
        
        full_matrix = torch.zeros((total_capacity, sample_dim), dtype=torch.float32, device=device)
        full_times = torch.zeros((total_capacity,), dtype=torch.float32, device=device)
        full_emotions = torch.zeros((total_capacity,), dtype=torch.float32, device=device)
        full_ids = [item['id'] for item in active_history] 

        if hist_len > 0:
            full_matrix[:hist_len] = torch.stack([item['vec'] for item in active_history])
            full_times[:hist_len] = torch.tensor([item['time'] for item in active_history], device=device)
            full_emotions[:hist_len] = torch.tensor([item['emotion'] for item in active_history], device=device)

        batch_edges = []      
        batch_csv_rows = []   

        # 3. 遍历 (Inner Loop)
        for i, (idx, row) in enumerate(w_t.iterrows()):
            tid = str(row['raw_value.id_str']).strip()
            uid = str(row['raw_value.user_id_str'])
            created_at = row['raw_value.created_at']
            ts = created_at.timestamp()

            raw_emo = get_raw_emotion(tid)
            
            G.add_node(tid, sentiment=float(raw_emo), opinion=float(raw_emo))
            tweet_to_user[tid] = uid
            
            if tid not in first_appearance:
                first_appearance[tid] = created_at

            curr_vec = get_combined_embedding(tid, raw_emo, vector_dict)
            if curr_vec is None: continue 

            updated_emotion = raw_emo
            
            # --- 建边计算 (基于无Tag语义+情感+时间衰减) ---
            current_pool_size = hist_len + i 
            if current_pool_size > 0:
                valid_matrix = full_matrix[:current_pool_size]
                valid_times = full_times[:current_pool_size]
                
                sims = torch.mv(valid_matrix, curr_vec)
                
                time_deltas = (ts - valid_times) / 3600.0
                tau = max(args.decay_unit_hours, 0.01)
                daws = torch.exp(-torch.abs(time_deltas) / tau)
                final_scores = sims * daws
                
                mask = final_scores > args.score_threshold
                valid_indices = torch.nonzero(mask).squeeze(1)
                
                if valid_indices.numel() > 0:
                    total_edges_all_time += valid_indices.numel()
                    v_scores = final_scores[valid_indices]
                    v_sims = sims[valid_indices]
                    v_times = full_times[valid_indices]
                    v_emos = full_emotions[valid_indices]
                    
                    idx_list = valid_indices.cpu().tolist()
                    score_list = v_scores.cpu().tolist()
                    
                    for k, hist_idx in enumerate(idx_list):
                        src_id = full_ids[hist_idx]
                        weight = score_list[k]
                        batch_edges.append((src_id, tid, weight))
                        batch_csv_rows.append([src_id, tid, created_at, f"{weight:.4f}"])

                    updated_emotion = calculate_vol_update(v_sims, v_times, v_emos, ts, raw_emo, args.decay_unit_hours)

            # --- 填入历史 ---
            new_hist_vec = get_combined_embedding(tid, updated_emotion, vector_dict)
            insert_idx = hist_len + i
            full_matrix[insert_idx] = new_hist_vec
            full_times[insert_idx] = ts
            full_emotions[insert_idx] = updated_emotion
            full_ids.append(tid)
            
            active_history.append({
                'id': tid, 'vec': new_hist_vec, 'time': ts, 'emotion': updated_emotion
            })
            
            if uid not in users_emotion_history: users_emotion_history[uid] = {}
            if w not in users_emotion_history[uid]: users_emotion_history[uid][w] = []
            users_emotion_history[uid][w].append(updated_emotion)

        # ================= 批量执行 =================
        if batch_edges:
            G.add_weighted_edges_from(batch_edges)
        if batch_csv_rows:
            edge_writer.writerows(batch_csv_rows)
        # ============================================

        # --- D. 统计指标 (GPU 快速传输版) ---
        num_edges = G.number_of_edges()
        pr_val, mod_val, assort_val, dcprr_val, cnlr_val = 0, 0, 0, 0, 0
        prop_scope, comp_intensity, collab_intensity = 0, 0, 0
        
        # [关键修复] 社区字典
        communities_dict = {}

        if num_edges > 0:
            # ================= [尝试 GPU 计算] =================
            if HAS_GPU_GRAPH:
                try:
                    G_int = nx.convert_node_labels_to_integers(G, label_attribute='orig_id')
                    edges_data = list(G_int.edges(data='weight', default=1.0))
                    if not edges_data: raise ValueError("No edges")
                    src_col, dst_col, wgt_col = zip(*edges_data)
                    gdf = cudf.DataFrame()
                    gdf['source'] = cudf.Series(src_col, dtype='int32')
                    gdf['target'] = cudf.Series(dst_col, dtype='int32')
                    gdf['weight'] = cudf.Series(wgt_col, dtype='float32')
                    
                    # PageRank
                    G_gpu = cugraph.Graph(directed=True)
                    G_gpu.from_cudf_edgelist(gdf, source='source', destination='target', edge_attr='weight')
                    pr_val = float(cugraph.pagerank(G_gpu, alpha=0.85)['pagerank'].mean())
                    
                    # Louvain
                    G_gpu_und = cugraph.Graph(directed=False)
                    G_gpu_und.from_cudf_edgelist(gdf, source='source', destination='target', edge_attr='weight')
                    parts, mod_score = cugraph.louvain(G_gpu_und)
                    mod_val = float(mod_score)

                    # [关键修复] 提取 GPU 社区结构到字典
                    parts_df = parts.to_pandas()
                    int_to_orig = nx.get_node_attributes(G_int, 'orig_id')
                    for row in parts_df.itertuples(index=False):
                        orig_id = int_to_orig.get(row.vertex)
                        if orig_id is not None:
                            communities_dict.setdefault(row.partition, []).append(orig_id)

                    # DCPRR
                    if DCPRR_IMPORTED and communities_dict:
                        dcprr_scores = calculate_all_dcprr_scores(G, communities_dict)
                        if dcprr_scores:
                            dcprr_val = float(np.mean(list(dcprr_scores.values())))
                                
                except Exception as e:
                    tqdm.write(f"\033[91m[Warning] GPU 加速失败: {e}\033[0m")
            
            # ================= [CPU 兜底] =================
            if mod_val == 0:
                try:
                    if pr_val == 0:
                        pr_val = float(np.mean(list(nx.pagerank(G, alpha=0.85).values())))
                    
                    import networkx.algorithms.community as nx_comm
                    G_und = G.to_undirected()
                    comm_sets = nx_comm.louvain_communities(G_und, seed=42)
                    mod_val = nx_comm.modularity(G_und, comm_sets)
                    
                    # [关键修复] CPU 社区字典
                    communities_dict = {i: list(c) for i, c in enumerate(comm_sets)}
                    
                    if DCPRR_IMPORTED and communities_dict:
                        dcprr_scores = calculate_all_dcprr_scores(G, communities_dict)
                        if dcprr_scores:
                            dcprr_val = float(np.mean(list(dcprr_scores.values())))
                except Exception as e_cpu:
                    pass

            # [关键修复] 调用 CNLR 计算逻辑
            try:
                current_edges_snapshot = set(G.edges())
                if prev_edges_snapshot and communities_dict:
                    cnlr_val = calculate_cnlr(current_edges_snapshot, prev_edges_snapshot, communities_dict)
                prev_edges_snapshot = current_edges_snapshot
            except: pass

            try: assort_val = nx.degree_assortativity_coefficient(G)
            except: assort_val = 0
            try: comp_intensity = float(sum(d.get('weight', 1.0) for _,_,d in G.edges(data=True)))
            except: comp_intensity = 0
            try:
                if num_edges < 1000000: 
                    G_und = G.to_undirected()
                    collab_intensity = int(sum(nx.triangles(G_und).values())/3)
            except: collab_intensity = 0
            try:
                comp_func = nx.weakly_connected_components if nx.is_directed(G) else nx.connected_components
                prop_scope = max((len(c) for c in comp_func(G)), default=0)
            except: prop_scope = 0

        stats_buffer.append([
            w.strftime('%Y-%m-%d'),
            G.number_of_nodes(), num_edges, pr_val, assort_val, mod_val, dcprr_val, cnlr_val, prop_scope, comp_intensity, collab_intensity
        ])

        if num_edges > 0:
            tqdm.write(f"[{w.strftime('%Y-%m-%d')}] N:{G.number_of_nodes()} E:{num_edges} Mod:{mod_val:.3f} DCPRR:{dcprr_val:.5f} CNLR:{cnlr_val:.5f}")
        else:
            tqdm.write(f"[{w.strftime('%Y-%m-%d')}] No Edges")
        
        if len(stats_buffer) >= 1:
            with open(index_path, 'a+', newline='') as f:
                wr = csv.writer(f)
                if f.tell() == 0: wr.writerow(['Time', 'Nodes', 'Edges', 'AvgPageRank', 'Assortativity', 'Modularity', 'DCPRR', 'CNLR', 'PropScope', 'CompIntensity', 'CollabIntensity'])
                wr.writerows(stats_buffer)
            stats_buffer = []

        timestamp_str = w.strftime('%Y-%m-%d_%H%M') 
        nx.write_edgelist(G, os.path.join(OUTPUT_DIR, f"tweet_network-{timestamp_str}.edgelist"), data=['weight'])
        if args.save_matrix and G.number_of_nodes() > 0:
            try:
                adj_matrix = nx.adjacency_matrix(G, weight='weight').todense()
                pd.DataFrame(adj_matrix, index=list(G.nodes()), columns=list(G.nodes())).to_csv(os.path.join(OUTPUT_DIR, f"G-implicit-{timestamp_str}.csv"))
            except: pass

    edge_log_file.close()
    
    with open(os.path.join(OUTPUT_DIR, "tweet_to_user.json"), "w") as f:
        json.dump(tweet_to_user, f)
        
    with open(os.path.join(OUTPUT_DIR, "first_appearance.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Node", "FirstAppearance"])
        for node, ts_val in first_appearance.items(): writer.writerow([node, ts_val])

    print("\n" + "="*60)
    print("               ABLATION (NO TAGS) REPORT")
    print("="*60)
    print(f"Total Edges Generated : {total_edges_all_time}")
    
    if os.path.exists(index_path):
        print("-" * 60)
        print("               GLOBAL STATISTICS SUMMARY (AVERAGE)")
        print("-" * 60)
        try:
            df_stats = pd.read_csv(index_path)
            numeric_cols = df_stats.select_dtypes(include=[np.number]).columns
            means = df_stats[numeric_cols].mean()
            for col, val in means.items(): print(f"{col:<20} | {val:.6f}")
            with open(index_path, 'a', newline='') as f:
                writer = csv.writer(f)
                row = ['GLOBAL_AVERAGE'] + [means.get(c, 0) for c in ['Nodes', 'Edges', 'AvgPageRank', 'Assortativity', 'Modularity', 'DCPRR', 'CNLR', 'PropScope', 'CompIntensity', 'CollabIntensity']]
                writer.writerow(row)
        except Exception as e: print(f"[Error] {e}")
    print("="*60 + "\n")

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