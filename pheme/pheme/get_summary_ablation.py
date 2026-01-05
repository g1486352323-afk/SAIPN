import pandas as pd
import os
import glob

# ================= 配置 =================
# 你的输出目录路径
BASE_OUTPUT_ROOT = "/data_huawei/gaohaizhen/network/saipn/model/ablation/outputs"

def summarize_results():
    print(f"📂 正在扫描目录: {BASE_OUTPUT_ROOT} ...\n")
    
    summary_list = []
    
    # 遍历 output 目录下的所有子文件夹
    subdirs = [d for d in os.listdir(BASE_OUTPUT_ROOT) if os.path.isdir(os.path.join(BASE_OUTPUT_ROOT, d))]
    subdirs.sort()

    for folder_name in subdirs:
        csv_path = os.path.join(BASE_OUTPUT_ROOT, folder_name, "index_gpu.csv")
        
        if os.path.exists(csv_path):
            try:
                # 读取 CSV
                df = pd.read_csv(csv_path)
                
                # -------------------------------------------------------
                # [核心修复] 兼容所有可能的标签名
                # -------------------------------------------------------
                # 1. 尝试找 GLOBAL_AVG
                row = df[df.iloc[:, 0] == 'GLOBAL_AVG']
                
                # 2. 如果没找到，尝试找 GLOBAL_AVERAGE
                if row.empty:
                    row = df[df.iloc[:, 0] == 'GLOBAL_AVERAGE']
                
                # 3. 还没找到，尝试找 GLOBAL_AVERAGE_WEIGHTED
                if row.empty:
                    row = df[df.iloc[:, 0] == 'GLOBAL_AVERAGE_WEIGHTED']
                
                # 4. 如果还是空，直接取最后一行 (兜底策略)
                if row.empty and len(df) > 0:
                    row = df.iloc[[-1]]

                # -------------------------------------------------------
                
                if not row.empty:
                    data = row.iloc[0].to_dict()
                    data['Exp_Name'] = folder_name
                    
                    # 尝试从文件夹名字里解析参数 (可选)
                    # 格式如: B_Win_002h-gpu
                    
                    # 清理不必要的列
                    if 'Time' in data: del data['Time']
                    # 删除标签列本身
                    keys_to_remove = [k for k in data.keys() if 'GLOBAL' in str(data[k])]
                    for k in keys_to_remove: del data[k]
                    
                    summary_list.append(data)
                else:
                    print(f"⚠️  {folder_name}: CSV 为空或格式错误")
            
            except Exception as e:
                print(f"❌ {folder_name}: 读取错误 - {e}")
        else:
            # print(f"   (跳过) {folder_name}: 无 index_gpu.csv")
            pass

    if summary_list:
        df_final = pd.DataFrame(summary_list)
        
        # 智能列排序
        first_cols = ['Exp_Name', 'Nodes', 'Edges', 'Modularity', 'DCPRR', 'Assortativity']
        cols = [c for c in first_cols if c in df_final.columns] + \
               [c for c in df_final.columns if c not in first_cols]
        df_final = df_final[cols]

        # 保存汇总表
        save_path = os.path.join(BASE_OUTPUT_ROOT, "final_summary_gpu.csv")
        df_final.to_csv(save_path, index=False)
        
        print("-" * 100)
        print(f"✅ 汇总成功！共找到 {len(df_final)} 条实验记录。")
        print(f"📄 文件已保存至: {save_path}")
        print("-" * 100)
        
        # 打印预览
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df_final.to_string())
    else:
        print("\n❌ 未找到任何有效数据，请检查路径是否正确：")
        print(f"   ls {BASE_OUTPUT_ROOT}")

if __name__ == "__main__":
    summarize_results()