import pandas as pd
import os
import glob

# ================= 配置 =================
BASE_DIR = '/data_huawei/gaohaizhen/network/saipn/model/ablation-d2/outputs'

# 定义实验名称和对应的文件夹路径
experiments = {
    "Baseline (Full)": "implicit-ablation-D",        # 完整版 (根据你之前的日志，这是baseline的路径)
    "No Sentiment":    "ablation_no_sentiment-D",    # 无情感
    "No Time Decay":   "ablation_no_time_decay-D",   # 无时间衰减
    "No Tags":         "ablation_no_tags-D"          # 无标签 (等你跑完这个就有数据了)
}
# =======================================

def summarize_all():
    print(f"📊 开始汇总消融实验结果 (Root: {BASE_DIR})...\n")
    
    summary_data = []
    
    for label, folder_name in experiments.items():
        csv_path = os.path.join(BASE_DIR, folder_name, "index_gpu.csv")
        
        if not os.path.exists(csv_path):
            print(f"❌ [Missing] {label}: 找不到文件 {csv_path}")
            # 添加一行空数据占位，方便知道缺了哪个
            empty_row = {"Experiment": label, "Status": "Missing/Failed"}
            summary_data.append(empty_row)
            continue
            
        try:
            # 读取 CSV
            df = pd.read_csv(csv_path)
            
            # 查找加权平均行 (优先找 WEIGHTED，兼容旧版 AVERAGE)
            row = df[df.iloc[:, 0] == 'GLOBAL_AVERAGE_WEIGHTED']
            if row.empty:
                row = df[df.iloc[:, 0] == 'GLOBAL_AVERAGE']
                
            if not row.empty:
                # 提取数据转换为字典
                data_dict = row.iloc[0].to_dict()
                
                # 清理不需要的列
                # 移除 Time 列
                keys_to_remove = [k for k in data_dict.keys() if 'Time' in k]
                # 移除第一列 (也就是值为 GLOBAL_AVERAGE_... 的那一列)
                keys_to_remove += [k for k, v in data_dict.items() if isinstance(v, str) and 'AVERAGE' in v]
                
                for k in keys_to_remove:
                    if k in data_dict: del data_dict[k]
                
                # 添加实验标签
                final_dict = {"Experiment": label}
                final_dict.update(data_dict)
                summary_data.append(final_dict)
                print(f"✅ [Loaded]  {label}")
            else:
                print(f"⚠️ [Empty]   {label}: 文件存在但没有汇总行")
                
        except Exception as e:
            print(f"❌ [Error]   {label}: 读取出错 - {e}")

    # --- 生成最终表格 ---
    if summary_data:
        df_final = pd.DataFrame(summary_data)
        
        # 重新排序列，把重要的指标放在前面
        # 假设常见的指标列名如下，根据你的csv实际列名会自动调整
        priority_cols = ["Experiment", "Modularity", "DCPRR", "Nodes", "Edges", "Assortativity", "CNLR", "CompIntensity"]
        existing_cols = [c for c in priority_cols if c in df_final.columns]
        other_cols = [c for c in df_final.columns if c not in priority_cols]
        
        df_final = df_final[existing_cols + other_cols]
        
        # 格式化数字 (保留4位小数)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print("\n" + "="*80)
        print("🚀 ABLATION STUDY SUMMARY TABLE")
        print("="*80)
        print(df_final.to_string(index=False))
        print("="*80)
        
        # 保存
        out_path = os.path.join(BASE_DIR, "final_ablation_summary-D.csv")
        df_final.to_csv(out_path, index=False)
        print(f"\n📄 汇总文件已保存: {out_path}")
    else:
        print("\n未提取到任何有效数据。")

if __name__ == "__main__":
    summarize_all()
    