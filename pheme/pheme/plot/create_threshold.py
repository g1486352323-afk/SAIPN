import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import LogLocator, NullLocator
import io

# ================= 1. 数据准备 (使用您提供的最新修正数据) =================
data_str = """
Index         Exp_Name  Threshold  Window  Decay       Nodes         Edges  Modularity   DCPRR  AvgPageRank  Assortativity      CNLR  CompIntensity   PropScope  CollabIntensity
0         B_Win_002h-gpu       0.70     2.0    1.0    200.8226  9.500000e+01      0.7759  0.8717       0.0017         0.4924  0.3163        72.4002    229.4189         186.4677
1         B_Win_006h-gpu       0.70     6.0    3.0    476.7742  7.828548e+02      0.7552  0.8068       0.0009         0.4134  0.2779       586.4219    1731.1304         1066.3387
2         B_Win_012h-gpu       0.70    12.0    6.0    904.0645  3.196984e+03      0.6680  0.7785       0.0007         0.1206  0.2133      2370.1224    2597.6369         4744.6129
3      B_Win_024h_1d-gpu       0.70    24.0   12.0   1758.6452  1.431971e+04      0.5746  0.7828       0.0004         0.0843  0.1153      10533.8971    3869.6634        10946.3226
4      B_Win_048h_2d-gpu       0.70    48.0   24.0   3467.8065  6.048585e+04      0.5396  0.8371       0.0003         0.0003  0.0759      44341.4979    6627.6679          577.7581
5      B_Win_072h_3d-gpu       0.70    72.0   36.0   5176.8387  1.298830e+05      0.5580  0.8120       0.0002         0.0002  0.0432      95149.8202   10233.0853          621.5484
6      B_Win_096h_4d-gpu       0.70    96.0   48.0   6884.7097  2.148172e+05      0.5699  0.8183       0.0001         0.0001  0.0358     157340.4602   11817.0026          651.3387
7      B_Win_120h_5d-gpu       0.70   120.0   60.0   8592.3226  3.111039e+05      0.5770  0.8346       0.0001         0.0001  0.0282     227843.9502   12852.2066          676.8548
8      B_Win_144h_6d-gpu       0.70   144.0   72.0  10299.0645  4.163365e+05      0.5810  0.8327       0.0001         0.0001  0.0193     304897.1658   13613.4743          706.2742
9      B_Win_168h_7d-gpu       0.70   168.0   84.0  12005.0645  5.285900e+05      0.5835  0.8157       0.0001         0.0001  0.0152     387097.9492   14198.7135          735.3548
10      C_Decay_002h-gpu       0.70   168.0    2.0  12005.0645  1.412153e+04      0.8660  0.9498       0.0001         0.5324  0.0488      10624.1877    3001.7723        15521.1290
11      C_Decay_006h-gpu       0.70   168.0    6.0  12005.0645  4.405566e+04      0.7459  0.8858       0.0001         0.0336  0.0409      32643.9637    4440.9831        16038.6774
12      C_Decay_012h-gpu       0.70   168.0   12.0  12005.0645  9.987852e+04      0.6419  0.8739       0.0001         0.0108  0.0240      73460.9723    5504.7895         4592.8871
13      C_Decay_024h-gpu       0.70   168.0   24.0  12005.0645  2.117648e+05      0.6010  0.8670       0.0001         0.0001  0.0219     155234.1642    9242.5371          592.4516
14      C_Decay_048h-gpu       0.70   168.0   48.0  12005.0645  3.763957e+05      0.5910  0.8309       0.0001         0.0001  0.0240     275680.2127   13362.3952          656.1129
15      C_Decay_072h-gpu       0.70   168.0   72.0  12005.0645  4.859926e+05      0.5858  0.8271       0.0001         0.0001  0.0170     355905.7523   13979.8138          706.2742
16      C_Decay_096h-gpu       0.70   168.0   96.0  12005.0645  5.660503e+05      0.5803  0.8075       0.0001         0.0001  0.0151     414524.7748   14361.1596          758.6290
17      C_Decay_120h-gpu       0.70   168.0  120.0  12005.0645  6.304164e+05      0.5749  0.7633       0.0001         0.0001  0.0157     461611.3627   14620.6989          798.7097
18      C_Decay_144h-gpu       0.70   168.0  144.0  12005.0645  6.871440e+05      0.5658  0.7526       0.0001         0.0001  0.0158     503052.6734   14841.2733          839.5484
19      C_Decay_168h-gpu       0.70   168.0  168.0  12005.0645  7.395490e+05      0.5526  0.7712       0.0001         0.0001  0.0130     541283.4795   15006.5526          891.0000
20  A_Th_0.65_Win24h-gpu       0.65    24.0   12.0   1758.6452  5.971237e+04      0.4785  0.7134       0.0004         0.0004  0.1040      40968.3745    5843.4518         1261.5806
21  A_Th_0.70_Win24h-gpu       0.70    24.0   12.0   1758.6452  1.431971e+04      0.5695  0.7871       0.0004         0.0843  0.1208      10533.8971    3869.6634        10946.3226
22  A_Th_0.75_Win24h-gpu       0.75    24.0   12.0   1758.6452  3.108613e+03      0.6775  0.8937       0.0004         0.3936  0.1402       2467.5148    1888.8790         3931.2258
23  A_Th_0.80_Win24h-gpu       0.80    24.0   12.0   1758.6452  8.042742e+02      0.8283  0.9524       0.0003         0.5055  0.1770        695.4852     529.0938         3147.3226
24  A_Th_0.85_Win24h-gpu       0.85    24.0   12.0   1758.6452  3.376613e+02      0.8984  0.9842       0.0003         0.6274  0.1790        313.2894      85.1615         1121.4032
25  A_Th_0.90_Win24h-gpu       0.90    24.0   12.0   1758.6452  2.106935e+02      0.9024  0.9930       0.0003         0.6166  0.1959        202.9244      27.5008          687.9839
26  A_Th_0.95_Win24h-gpu       0.95    24.0   12.0   1758.6452  1.381935e+02      0.8772  0.9867       0.0003         0.5903  0.1872        135.9365      20.8209          434.4194
27  A_Th_0.50_Win24h-gpu       0.50    24.0   12.0   1758.6452  1.084631e+06      0.2477  0.7037       0.0005        -0.0001  0.1115     608069.1426    8170.7235         1678.2419
28  A_Th_0.55_Win24h-gpu       0.55    24.0   12.0   1758.6452  5.325740e+05      0.3027  0.6915       0.0005         0.0001  0.1085     318646.0871    7752.2702        11060.9839
29  A_Th_0.60_Win24h-gpu       0.60    24.0   12.0   1758.6452  2.013080e+05      0.3806  0.5755       0.0004         0.0001  0.0918     128960.7639    7107.8920         4675.2258
"""

# 读取数据，处理第一列为 Index
df_all = pd.read_csv(io.StringIO(data_str), sep=r'\s+', index_col=0)

# 数据分组与预处理
df_a = df_all[df_all['Exp_Name'].str.startswith('A')].copy().sort_values('Threshold').reset_index(drop=True)
df_b = df_all[df_all['Exp_Name'].str.startswith('B')].copy().sort_values('Window').reset_index(drop=True)
df_c = df_all[df_all['Exp_Name'].str.startswith('C')].copy().sort_values('Decay').reset_index(drop=True)

# ================= 2. 绘图函数 =================

def create_equidistant_plot(df, x_col, x_label_name, optimal_val, optimal_text, filename):
    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set(style="ticks", font_scale=1.5)
    fig, ax1 = plt.subplots(figsize=(18, 12))
    
    # 使用索引作为X轴坐标，实现等间距效果
    x_indices = np.arange(len(df))
    lines = []

    # === 左轴 (Linear Scale, 0 ~ 1.25) ===
    # Modularity (0-1)
    l1, = ax1.plot(x_indices, df['Modularity'], marker='o', markersize=10, color='#2ca02c', lw=3.5, label='Modularity')
    # DCPRR (0-1)
    l2, = ax1.plot(x_indices, df['DCPRR'], marker='s', markersize=10, color='#1f77b4', lw=3, linestyle='--', label='DCPRR')
    # Assortativity (0-1)
    l3, = ax1.plot(x_indices, df['Assortativity'], marker='^', markersize=10, color='#ff7f0e', lw=3, linestyle='-.', label='Assortativity')
    # AvgPageRank (数值很小, 乘以100以便展示)
    l5, = ax1.plot(x_indices, df['AvgPageRank'] * 100, marker='P', markersize=10, color='#e377c2', lw=2.5, linestyle='-', label='AvgPageRank (x100)')
    # CNLR (数值在 0.01 ~ 0.31 之间) -> 放在左轴
    l4, = ax1.plot(x_indices, df['CNLR'], marker='*', markersize=11, color='#7f7f7f', lw=2.5, linestyle=':', label='CNLR')
    
    lines.extend([l1, l2, l3, l4, l5])

    # 设置左轴标签和刻度
    ax1.set_xlabel(x_label_name, fontsize=20, fontweight='bold', labelpad=15)
    ax1.set_xticks(x_indices)
    
    # 格式化 X 轴标签：如果是整数显示整数，否则显示浮点数
    def fmt_label(x): 
        return f"{int(x)}" if x == int(x) else f"{x}"
    ax1.set_xticklabels([fmt_label(val) for val in df[x_col]], fontsize=16)

    ax1.set_ylabel('Structural Quality Coefficients (Linear)', fontsize=20, fontweight='bold', color='#333333', labelpad=20)
    ax1.set_ylim(0, 1.25)
    ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
    ax1.tick_params(axis='y', labelcolor='#333333', labelsize=16)
    ax1.grid(True, which='major', linestyle='--', alpha=0.5, linewidth=1)

    # === 右轴 (Log Scale) ===
    ax2 = ax1.twinx()
    # Edges (10^2 ~ 10^6)
    l6, = ax2.plot(x_indices, df['Edges'], marker='D', markersize=10, color='#d62728', lw=3.5, label='Edges (Log)')
    # PropScope (10^2 ~ 10^4)
    l7, = ax2.plot(x_indices, df['PropScope'], marker='X', markersize=11, color='#9467bd', lw=3, linestyle='--', markeredgewidth=2, label='Nodes/PropScope (Log)') 
    # CollabIntensity (10^2 ~ 10^4)
    l8, = ax2.plot(x_indices, df['CollabIntensity'], marker='p', markersize=11, color='#8c564b', lw=3, linestyle='-', label='CollabIntensity (Log)')
    # CompIntensity (数值极大, 70 ~ 600,000) -> 放在右轴
    l9, = ax2.plot(x_indices, df['CompIntensity'], marker='v', markersize=11, color='#17becf', lw=3, linestyle='-', label='CompIntensity (Log)')
    
    lines.extend([l6, l7, l8, l9])

    # 设置右轴标签和刻度
    ax2.set_ylabel('Network Magnitude (Log Scale)', fontsize=20, fontweight='bold', color='black', labelpad=15)
    ax2.tick_params(axis='y', labelcolor='black', labelsize=16, which='both', width=1.5)
    ax2.set_yscale('log')
    ax2.set_ylim(10**1.5, 10**7.2) 
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
    ax2.yaxis.set_minor_locator(NullLocator()) 

    # === 绘制 Optimal 虚线 ===
    if optimal_val is not None:
        opt_idx_list = df.index[df[x_col] == optimal_val].tolist()
        if opt_idx_list:
            opt_idx = opt_idx_list[0]
            plt.axvline(x=opt_idx, color='black', linestyle=':', alpha=0.6, lw=2.5)
            ax1.text(opt_idx, 1.18, optimal_text, ha='center', va='top', fontsize=16, fontweight='bold', color='black', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    # === 图例 ===
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.0, 1.01),  
               title="Performance Metrics", title_fontsize=16, fontsize=13, 
               ncol=2, frameon=True, framealpha=0.9, shadow=True, borderpad=1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {filename}")

# ================= 3. 生成图表 =================

# 参数设定：阈值 0.70, 窗口 24h, 衰减 2h
create_equidistant_plot(df_a, 'Threshold', r'Similarity Threshold ($\theta$)', 0.70, 'Optimal\n$\\theta=0.70$', 'plot_A_final.png')
create_equidistant_plot(df_b, 'Window', 'Sliding Window Size (Hours)', 24.0, 'Optimal Window\n~24 Hours', 'plot_B_final.png')
create_equidistant_plot(df_c, 'Decay', 'Decay Half-Life (Hours)', 2.0, 'Optimal Decay\n~2 Hours', 'plot_C_final.png')