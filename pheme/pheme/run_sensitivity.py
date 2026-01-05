import os
import pandas as pd
import time
import glob

# ================= 配置 =================
# 请确保指向修改后的 full.py
PYTHON_SCRIPT = "/data_huawei/gaohaizhen/network/saipn/model/ablation/full.py"
BASE_OUTPUT_ROOT = "/data_huawei/gaohaizhen/network/saipn/model/ablation/outputs"

# ================= 实验计划 =================
experiments = []



# --- Group B: 窗口规模敏感性 (2h -> 168h) ---
experiments.extend([
    {"th": 0.70, "decay": 1.0,   "win": 2.0,    "name": "B_Win_002h-gpu"}, 
    {"th": 0.70, "decay": 3.0,   "win": 6.0,    "name": "B_Win_006h-gpu"}, 
    {"th": 0.70, "decay": 6.0,   "win": 12.0,   "name": "B_Win_012h-gpu"}, 
    {"th": 0.70, "decay": 12.0,  "win": 24.0,   "name": "B_Win_024h_1d-gpu"}, 
    {"th": 0.70, "decay": 24.0,  "win": 48.0,   "name": "B_Win_048h_2d-gpu"}, 
    {"th": 0.70, "decay": 36.0,  "win": 72.0,   "name": "B_Win_072h_3d-gpu"}, 
    {"th": 0.70, "decay": 48.0,  "win": 96.0,   "name": "B_Win_096h_4d-gpu"}, 
    {"th": 0.70, "decay": 60.0,  "win": 120.0,  "name": "B_Win_120h_5d-gpu"}, 
    {"th": 0.70, "decay": 72.0,  "win": 144.0,  "name": "B_Win_144h_6d-gpu"}, 
    {"th": 0.70, "decay": 84.0,  "win": 168.0,  "name": "B_Win_168h_7d-gpu"}, 
])

# --- Group C: 衰减敏感性 (固定窗口 168h) ---
FIXED_WIN_LONG = 168.0 
experiments.extend([
    {"th": 0.70, "decay": 2.0,   "win": FIXED_WIN_LONG, "name": "C_Decay_002h-gpu"}, 
    {"th": 0.70, "decay": 6.0,   "win": FIXED_WIN_LONG, "name": "C_Decay_006h-gpu"}, 
    {"th": 0.70, "decay": 12.0,  "win": FIXED_WIN_LONG, "name": "C_Decay_012h-gpu"}, 
    {"th": 0.70, "decay": 24.0,  "win": FIXED_WIN_LONG, "name": "C_Decay_024h-gpu"}, 
    {"th": 0.70, "decay": 48.0,  "win": FIXED_WIN_LONG, "name": "C_Decay_048h-gpu"}, 
    {"th": 0.70, "decay": 72.0,  "win": FIXED_WIN_LONG, "name": "C_Decay_072h-gpu"}, 
    {"th": 0.70, "decay": 96.0,  "win": FIXED_WIN_LONG, "name": "C_Decay_096h-gpu"}, 
    {"th": 0.70, "decay": 120.0, "win": FIXED_WIN_LONG, "name": "C_Decay_120h-gpu"}, 
    {"th": 0.70, "decay": 144.0, "win": FIXED_WIN_LONG, "name": "C_Decay_144h-gpu"}, 
    {"th": 0.70, "decay": 168.0, "win": FIXED_WIN_LONG, "name": "C_Decay_168h-gpu"}, 
])


# --- Group A: 阈值敏感性 ---
experiments.extend([

    {"th": 0.65, "decay": 12.0, "win": 24.0, "name": "A_Th_0.65_Win24h-gpu"},
    {"th": 0.70, "decay": 12.0, "win": 24.0, "name": "A_Th_0.70_Win24h-gpu"}, 
    {"th": 0.75, "decay": 12.0, "win": 24.0, "name": "A_Th_0.75_Win24h-gpu"},
    {"th": 0.80, "decay": 12.0, "win": 24.0, "name": "A_Th_0.80_Win24h-gpu"},
    {"th": 0.85, "decay": 12.0, "win": 24.0, "name": "A_Th_0.85_Win24h-gpu"},
    {"th": 0.90, "decay": 12.0, "win": 24.0, "name": "A_Th_0.90_Win24h-gpu"},
    {"th": 0.95, "decay": 12.0, "win": 24.0, "name": "A_Th_0.95_Win24h-gpu"},
    {"th": 0.50, "decay": 12.0, "win": 24.0, "name": "A_Th_0.50_Win24h-gpu"},
    {"th": 0.55, "decay": 12.0, "win": 24.0, "name": "A_Th_0.55_Win24h-gpu"},
    {"th": 0.60, "decay": 12.0, "win": 24.0, "name": "A_Th_0.60_Win24h-gpu"},
])
# 去重
seen = set()
final_exps = []
for exp in experiments:
    if exp['name'] not in seen:
        final_exps.append(exp)
        seen.add(exp['name'])
experiments = final_exps

def run_batch():
    print(f"🚀 开始 Charlie Hebdo 寻参，共 {len(experiments)} 组实验...\n")
    print(f"⚠️  模式：每小时演化 | 每6小时采样 | 强制覆盖 (--force)\n")
    
    for i, exp in enumerate(experiments):
        out_dir = os.path.join(BASE_OUTPUT_ROOT, exp["name"])
        
        print(f"[{i+1}/{len(experiments)}] Running: {exp['name']}")
        print(f"   Param: Th={exp['th']}, Decay={exp['decay']}h, Win={exp['win']}h")
        
        if not os.path.exists(BASE_OUTPUT_ROOT):
            os.makedirs(BASE_OUTPUT_ROOT, exist_ok=True)

        cmd = (
            f"python {PYTHON_SCRIPT} "
            f"--score-threshold {exp['th']} "
            f"--decay-unit-hours {exp['decay']} "
            f"--delete-after-hours {exp['win']} "
            f"--output-dir {out_dir} "
            f"--resample H " 
            f"--force"  # <--- 强制覆盖
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
        csv_path = os.path.join(BASE_OUTPUT_ROOT, exp["name"], "index_gpu.csv")
        
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # 查找汇总行
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
                    keys_to_remove = [k for k, v in data.items() if isinstance(v, str) and 'AVERAGE' in v]
                    for k in keys_to_remove: del data[k]
                    
                    summary_list.append(data)
            except Exception as e:
                print(f"❌ 读取错误 {exp['name']}: {e}")
        else:
            pass
    
    if summary_list:
        df_final = pd.DataFrame(summary_list)
        cols_order = ['Exp_Name', 'Threshold', 'Window', 'Decay', 'Nodes', 'Edges', 'Modularity', 'DCPRR']
        cols = [c for c in cols_order if c in df_final.columns] + [c for c in df_final.columns if c not in cols_order]
        df_final = df_final[cols]
        
        save_path = os.path.join(BASE_OUTPUT_ROOT, "sensitivity_summary_charlie_hybrid_sampled.csv")
        df_final.to_csv(save_path, index=False)
        print(f"✨ 汇总表格已生成: {save_path}")
        print("-" * 120)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_final.round(4).to_string())
    else:
        print("未找到任何有效结果。")

if __name__ == "__main__":
    run_batch()
    summarize()