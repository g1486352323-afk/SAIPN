import os
import pandas as pd
import time
import glob

# ================= 配置 =================
PYTHON_SCRIPT = "/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/full.py"  
BASE_OUTPUT_ROOT = "/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/outputs"

# ================= 实验计划：全方位敏感性分析 =================
experiments = []

# --- Group A: 阈值敏感性 (已跑完，代码保留用于汇总) ---
experiments.extend([
    {"th": 0.50, "decay": 360.0, "win": 720.0, "name": "A_Th_0.50_Win30d"},
    {"th": 0.55, "decay": 360.0, "win": 720.0, "name": "A_Th_0.55_Win30d"}, 
    {"th": 0.60, "decay": 360.0, "win": 720.0, "name": "A_Th_0.60_Win30d"}, 
    {"th": 0.65, "decay": 360.0, "win": 720.0, "name": "A_Th_0.65_Win30d"}, 
    {"th": 0.70, "decay": 360.0, "win": 720.0, "name": "A_Th_0.70_Win30d"}, 
    {"th": 0.75, "decay": 360.0, "win": 720.0, "name": "A_Th_0.75_Win30d"},
    {"th": 0.80, "decay": 360.0, "win": 720.0, "name": "A_Th_0.80_Win30d"},
    {"th": 0.85, "decay": 360.0, "win": 720.0, "name": "A_Th_0.85_Win30d"}, 
    {"th": 0.90, "decay": 360.0, "win": 720.0, "name": "A_Th_0.90_Win30d"},
    {"th": 0.95, "decay": 360.0, "win": 720.0, "name": "A_Th_0.95_Win30d"}, 
])

# --- Group B: 窗口规模敏感性 (修正命名，去掉括号) ---
# [Fix] Removed '(' and ')' to prevent shell execution errors
experiments.extend([
    {"th": 0.70, "decay": 84.0,  "win": 168.0,  "name": "B_Win_07d_1wk"}, # Modified
    {"th": 0.70, "decay": 168.0, "win": 336.0,  "name": "B_Win_14d_2wk"}, # Modified
    {"th": 0.70, "decay": 252.0, "win": 504.0,  "name": "B_Win_21d_3wk"}, # Modified
    {"th": 0.70, "decay": 336.0, "win": 672.0,  "name": "B_Win_28d_4wk"}, # Modified
    {"th": 0.70, "decay": 420.0, "win": 840.0,  "name": "B_Win_35d_5wk"}, # Modified
    {"th": 0.70, "decay": 504.0, "win": 1008.0, "name": "B_Win_42d_6wk"}, # Modified
    {"th": 0.70, "decay": 588.0, "win": 1176.0, "name": "B_Win_49d_7wk"}, # Modified
    {"th": 0.70, "decay": 672.0, "win": 1344.0, "name": "B_Win_56d_8wk"}, # Modified
    {"th": 0.70, "decay": 756.0, "win": 1512.0, "name": "B_Win_63d_9wk"}, # Modified
    {"th": 0.70, "decay": 840.0, "win": 1680.0, "name": "B_Win_70d_10wk"},# Modified
])

# --- Group C: 衰减速度敏感性 (已跑完，代码保留用于汇总) ---
experiments.extend([
    {"th": 0.70, "decay": 48.0,   "win": 1440.0, "name": "C_Decay_02d_Win60d"},
    {"th": 0.70, "decay": 120.0,  "win": 1440.0, "name": "C_Decay_05d_Win60d"},
    {"th": 0.70, "decay": 168.0,  "win": 1440.0, "name": "C_Decay_07d_Win60d"},
    {"th": 0.70, "decay": 240.0,  "win": 1440.0, "name": "C_Decay_10d_Win60d"},
    {"th": 0.70, "decay": 336.0,  "win": 1440.0, "name": "C_Decay_14d_Win60d"},
    {"th": 0.70, "decay": 504.0,  "win": 1440.0, "name": "C_Decay_21d_Win60d"},
    {"th": 0.70, "decay": 720.0,  "win": 1440.0, "name": "C_Decay_30d_Win60d"},
    {"th": 0.70, "decay": 960.0,  "win": 1440.0, "name": "C_Decay_40d_Win60d"},
    {"th": 0.70, "decay": 1200.0, "win": 1440.0, "name": "C_Decay_50d_Win60d"},
    {"th": 0.70, "decay": 1440.0, "win": 1440.0, "name": "C_Decay_60d_Win60d"},
])


def run_batch():
    print(f"🚀 开始参数敏感性分析，共 {len(experiments)} 组实验...\n")
    
    for i, exp in enumerate(experiments):
        out_dir = os.path.join(BASE_OUTPUT_ROOT, exp["name"])
        result_file = os.path.join(out_dir, "index_gpu.csv")

        # [功能新增] 检查是否已经跑过
        if os.path.exists(result_file):
             print(f"[{i+1}/{len(experiments)}] ⏭️  跳过 (已存在): {exp['name']}")
             continue

        print(f"[{i+1}/{len(experiments)}] Running: {exp['name']}")
        print(f"   Param: Threshold={exp['th']}, Decay={exp['decay']}h, Window={exp['win']}h")
        
        # 确保目录存在
        os.makedirs(out_dir, exist_ok=True)

        cmd = (
            f"python {PYTHON_SCRIPT} "
            f"--score-threshold {exp['th']} "
            f"--decay-unit-hours {exp['decay']} "
            f"--delete-after-hours {exp['win']} "
            f"--output-dir {out_dir} "
            f"--resample D" 
        )
        
        start = time.time()
        ret = os.system(cmd)
        duration = time.time() - start
        
        if ret != 0:
            print(f"❌ 运行失败: {exp['name']}\n")
        else:
            print(f"✅ 运行完成: {exp['name']} (耗时: {duration:.1f}s)\n")

def summarize():
    print("\n📊 正在汇总所有结果...")
    summary_list = []
    
    for exp in experiments:
        # 注意：这里我们读取的是 index_gpu.csv
        csv_path = os.path.join(BASE_OUTPUT_ROOT, exp["name"], "index_gpu.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                
                # [Fix] 优先查找 WEIGHTED，如果没有再找普通 AVERAGE
                row = df[df.iloc[:, 0] == 'GLOBAL_AVERAGE_WEIGHTED']
                if row.empty:
                    row = df[df.iloc[:, 0] == 'GLOBAL_AVERAGE']
                
                if not row.empty:
                    data = row.iloc[0].to_dict()
                    data['Exp_Name'] = exp['name']
                    data['Threshold'] = exp['th']
                    data['Decay'] = exp['decay']
                    data['Window'] = exp['win']
                    
                    if 'Time' in data: del data['Time']
                    # 删除值为 GLOBAL_AVERAGE_... 的那一列
                    keys_to_remove = [k for k, v in data.items() if isinstance(v, str) and 'AVERAGE' in v]
                    for k in keys_to_remove: del data[k]
                    
                    summary_list.append(data)
                else:
                    print(f"⚠️  {exp['name']}: 只有表头，无汇总行")
            except Exception as e:
                print(f"❌ 读取错误 {exp['name']}: {e}")
        else:
            # Group B 如果还没跑完，这里会报找不到，属于正常
            # print(f"⚪ 尚未生成: {exp['name']}")
            pass
    
    if summary_list:
        df_final = pd.DataFrame(summary_list)
        
        # 智能整理列顺序
        desired_order = ['Exp_Name', 'Threshold', 'Window', 'Decay', 'Nodes', 'Edges', 'Modularity', 'DCPRR', 'AvgPageRank', 'CompIntensity']
        cols = [c for c in desired_order if c in df_final.columns] + [c for c in df_final.columns if c not in desired_order]
        df_final = df_final[cols]
        
        save_path = os.path.join(BASE_OUTPUT_ROOT, "sensitivity_summary.csv")
        df_final.to_csv(save_path, index=False)
        print(f"✨ 汇总表格已生成: {save_path}")
        print("-" * 120)
        # 设置显示格式，防止省略
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_final.round(4).to_string())
    else:
        print("未找到任何有效结果。")

if __name__ == "__main__":
    run_batch()
    summarize()