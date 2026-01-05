#!/usr/bin/env python3
import pandas as pd
import os

print("测试脚本开始")

# 检查文件
input_file = 'charliehebdo_gemini_2_flash_output_fixed_from_cleaned.csv'
print(f"检查文件: {input_file}")
print(f"文件存在: {os.path.exists(input_file)}")

if os.path.exists(input_file):
    print("读取数据...")
    df = pd.read_csv(input_file, nrows=5)
    print(f"列名: {list(df.columns)}")
    print(f"Tags列存在: {'Tags' in df.columns}")

    if 'Tags' in df.columns:
        print("Tags示例:")
        for i, tag in enumerate(df['Tags'].head()):
            print(f"  {i}: {tag}")

print("测试完成")


