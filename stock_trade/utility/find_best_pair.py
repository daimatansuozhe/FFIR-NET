import os
import pandas as pd
from datetime import datetime
from pair_trading import PairTradingStrategy
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1. 加载行业分类数据
industry_df = pd.read_csv("../zz500行业分类.csv", encoding='utf-8')
industry_df = industry_df.rename(columns={
    "证券代码": "stock_code",
    "证券简称": "stock_name",
    "上市日期": "list_date",
    "所属行业": "industry"
})

# 2. 加载股票价格数据
stock_folder = "data2"
stock_data = {}

for filename in os.listdir(stock_folder):
    if filename.endswith(".csv"):
        stock_code = filename.replace(".csv", "")
        file_path = os.path.join(stock_folder, filename)
        df = pd.read_csv(file_path, encoding='utf-8')
        df['时间'] = pd.to_datetime(df['时间'])
        df.set_index('时间', inplace=True)
        stock_data[stock_code] = df[['收盘价', '成交额.元.']]

# 获取所有有效的行业
valid_industries = industry_df['industry'].value_counts()[
    industry_df['industry'].value_counts() > 1].index.tolist()

# 初始化最佳组合信息
best_performance = {
    'annualized_return': float('-inf'),
    'max_drawdown': float('inf'),
    'sharpe_ratio': float('-inf'),  # 新增夏普率
    'selected_pairs': [],
    'industry_pair': ()
}

# 创建一个列表来保存所有回测结果
all_results = []

# 遍历所有可能的行业组合
for i in tqdm(range(len(valid_industries))):
    for j in tqdm(range(i + 1, len(valid_industries))):
        ind1 = valid_industries[i]
        ind2 = valid_industries[j]

        # 3. 初始化策略类
        strategy = PairTradingStrategy(stock_data, industry_df)

        # 4. 选择股票对（形成期：过去252个交易日）
        selected_pairs = strategy.select_pairs(ind1, ind2)

        if not selected_pairs:
            continue

        # 5. 回测策略（假设使用近一年交易数据）
        start_date = pd.to_datetime("2024-04-30")
        end_date = pd.to_datetime("2025-04-30")
        performance = strategy.backtest(start_date, end_date)

        # 将当前行业对的回测结果添加到总结果列表中
        for pair_info in selected_pairs:
            result = {
                'industry_pair': f"{ind1}-{ind2}",
                'stock_pair': pair_info['pair'],
                'spread_std': pair_info['spread_std'],
                'total_avg_volume': pair_info['total_avg_volume'],
                'annualized_return': performance['annualized_return'],
                'max_drawdown': performance['max_drawdown'],
                'sharpe_ratio': performance['sharpe_ratio']  # 新增夏普率
            }
            all_results.append(result)

        # 6. 检查是否为最佳组合（修改判断条件，加入夏普率）
        if (performance['annualized_return'] > best_performance['annualized_return']) or \
           (performance['annualized_return'] == best_performance['annualized_return'] and
            performance['max_drawdown'] < best_performance['max_drawdown']) or \
           (performance['annualized_return'] == best_performance['annualized_return'] and
            performance['max_drawdown'] == best_performance['max_drawdown'] and
            performance['sharpe_ratio'] > best_performance['sharpe_ratio']):
            best_performance = {
                'annualized_return': performance['annualized_return'],
                'max_drawdown': performance['max_drawdown'],
                'sharpe_ratio': performance['sharpe_ratio'],  # 新增夏普率
                'selected_pairs': selected_pairs,
                'industry_pair': (ind1, ind2)
            }

# 将所有结果保存为CSV文件
results_df = pd.DataFrame(all_results)
results_df.to_csv('pair_trading_results.csv', index=False, encoding='utf-8-sig')
print("所有股票对的回测结果已保存至 'pair_trading_results.csv'")

# 输出最佳组合信息（新增夏普率显示）
print("\n最佳行业组合：", best_performance['industry_pair'])
print("选出的股票对：")
for p in best_performance['selected_pairs']:
    print(f"股票对: {p['pair']}, 残差标准差: {p['spread_std']:.4f}, 日均成交额: {p['total_avg_volume']:.2f}")
print("\n最佳策略表现：")
print(f"年化收益率: {best_performance['annualized_return']:.4f}")
print(f"最大回撤: {best_performance['max_drawdown']:.4f}")
print(f"夏普比率: {best_performance['sharpe_ratio']:.4f}")  # 新增夏普率显示

# 7. 绘制组合价值变化图（使用最佳组合）
start_date = pd.to_datetime("2024-04-30")
end_date = pd.to_datetime("2025-04-30")
strategy = PairTradingStrategy(stock_data, industry_df)
strategy.select_pairs(*best_performance['industry_pair'])
strategy.backtest(start_date, end_date)
dates, values = zip(*strategy.portfolio_value)
plt.figure(figsize=(12, 6))
plt.plot(dates, values, label='Portfolio Value')
plt.title(f"配对交易策略组合价值变化（最佳组合，夏普率: {best_performance['sharpe_ratio']:.4f}）")
plt.xlabel("日期")
plt.ylabel("组合价值")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('portfolio_value_chart.png')  # 保存图表为图片文件
print("组合价值变化图已保存至 'portfolio_value_chart.png'")
# plt.show()  # 如果在服务器环境运行，可能无法显示图形，改为保存为文件