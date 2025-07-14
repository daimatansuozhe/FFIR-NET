import matplotlib.pyplot as plt
import pandas as pd

# 读取 Excel 文件
excel_file = pd.ExcelFile('堆叠图.xlsx')
# 获取指定工作表中的数据
df = excel_file.parse('Sheet1')
import matplotlib.pyplot as plt

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']

# 提取出各行业不合标股票数、合标股票数和股票数数据
new_df = pd.DataFrame({
    '行业': df.columns[1:],
    '不合标股票数': df.iloc[0, 1:],
    '合标股票数': df.iloc[1, 1:],
    '股票数': df.iloc[2, 1:]
})

# 按照股票数降序排列
new_df = new_df.sort_values(by='股票数', ascending=False)

# 设置行业为索引
new_df.set_index('行业', inplace=True)

# 绘制水平堆叠柱状图
ax = new_df[['不合标股票数', '合标股票数']].plot(kind='barh', stacked=True)

for container in ax.containers:
    for index, rect in enumerate(container):
        width = rect.get_width()
        y_coord = rect.get_y() + rect.get_height() / 2
        x_coord = rect.get_x() + width
        if width > 0:
            ax.text(x_coord, y_coord, str(int(width)), ha='left', va='center', fontsize=6)

# 设置标题和标签
plt.title('各行业股票数堆叠柱状图')
# plt.ylabel('行业')
# plt.xlabel('数量')
plt.xticks(fontsize=10)  # 设置x轴标签字体大小
plt.yticks(fontsize=8)  # 设置y轴标签字体大小
# 显示图形
plt.tight_layout()
save_path = "./fig/堆叠图"  # 修改为你的目标路径
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi控制清晰度，bbox_inches去除白边
