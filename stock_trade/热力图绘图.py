import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv('协整性距离.csv')

# 设置中文字体和负号
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

# 设置`Unnamed: 0`列为索引
df.set_index('Unnamed: 0', inplace=True)

# 构造严格上三角掩码（包含对角线）
mask = ~np.triu(np.ones(df.shape), k=1).astype(bool)  # 保留 k=1 以上的上三角（排除对角线）

# 应用掩码，仅保留严格上三角
df_masked = df.mask(mask)

# 绘制热力图
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(df_masked, cmap='YlOrRd', annot=True, fmt='.2f',
                      annot_kws={'size': 6}, cbar=True)

# 设置标题和坐标轴样式
plt.title('行业间协整性热力图', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)

# 保存图像
save_path = "./fig/行业间协整性热力图_严格上三角.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight')
