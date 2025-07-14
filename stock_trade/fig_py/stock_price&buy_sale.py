import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import numpy as np
import argparse


# 配对交易策略类
class PairTradingStrategy:
    def __init__(self, stock_data, industry_df, formation_period=252, z_threshold=1.0,
                 stop_loss=3.0, take_profit=0.1, initial_capital=1000000):
        self.stock_data = stock_data
        self.industry_df = industry_df
        self.formation_period = formation_period
        self.z_threshold = z_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_capital = initial_capital
        self.selected_pairs = []
        self.portfolio_value = []

    def _calculate_half_life(self, spread):
        """计算价差序列的半衰期"""
        # 使用AR(1)模型计算半衰期
        delta_spread = np.diff(spread)
        spread_lag = spread[:-1]

        # 添加常数项进行线性回归
        X = np.column_stack((np.ones_like(spread_lag), spread_lag))
        beta = np.linalg.lstsq(X, delta_spread, rcond=None)[0][1]

        # 计算半衰期
        half_life = -np.log(2) / beta
        return half_life

    def backtest(self, start_date, end_date):
        """执行回测"""
        # 初始化组合价值
        current_capital = self.initial_capital
        self.portfolio_value = [(start_date, current_capital)]

        # 获取回测期间的日期列表
        date_range = pd.date_range(start=start_date, end=end_date)

        # 为每个股票对初始化仓位和交易记录
        for pair_info in self.selected_pairs:
            pair_info['position'] = 0  # 0: 无仓位, 1: 做多股票1做空股票2, -1: 做空股票1做多股票2
            pair_info['entry_price'] = 0  # 入场价差
            pair_info['positions'] = []  # 记录所有交易信号
            pair_info['shares'] = (0, 0)  # 记录持仓数量，(股票1持仓, 股票2持仓)

        # 按日期进行回测
        for date in date_range:
            # 跳过非交易日
            if date not in self.stock_data[self.selected_pairs[0]['pair'][0]].index:
                continue

            # 更新每对股票的交易状态
            for pair_info in self.selected_pairs:
                pair = pair_info['pair']
                s1, s2 = pair

                # 获取当前价格
                s1_price = self.stock_data[s1].loc[date, '收盘价']
                s2_price = self.stock_data[s2].loc[date, '收盘价']

                # 计算当前价差和Z得分
                current_spread = s1_price - s2_price
                current_z_score = (current_spread - pair_info['spread_mean']) / pair_info['spread_std']

                # 获取当前仓位
                position = pair_info['position']

                # 交易逻辑
                if position == 0:  # 无仓位，检查是否开仓
                    # 开仓条件：Z得分超过阈值
                    if current_z_score > self.z_threshold:
                        # 做空股票1，做多股票2
                        pair_info['position'] = -1
                        pair_info['entry_price'] = current_spread
                        pair_info['positions'].append({
                            'date': date,
                            'position': -1,
                            's1_price': s1_price,
                            's2_price': s2_price,
                            'z_score': current_z_score
                        })
                        # 计算持仓数量，假设平均分配资金
                        half_capital = current_capital / 2
                        shares_s1 = -half_capital / s1_price
                        shares_s2 = half_capital / s2_price
                        pair_info['shares'] = (shares_s1, shares_s2)
                    elif current_z_score < -self.z_threshold:
                        # 做多股票1，做空股票2
                        pair_info['position'] = 1
                        pair_info['entry_price'] = current_spread
                        pair_info['positions'].append({
                            'date': date,
                            'position': 1,
                            's1_price': s1_price,
                            's2_price': s2_price,
                            'z_score': current_z_score
                        })
                        # 计算持仓数量，假设平均分配资金
                        half_capital = current_capital / 2
                        shares_s1 = half_capital / s1_price
                        shares_s2 = -half_capital / s2_price
                        pair_info['shares'] = (shares_s1, shares_s2)
                else:  # 有仓位，检查是否平仓
                    entry_spread = pair_info['entry_price']
                    spread_change = (current_spread - entry_spread) / entry_spread

                    # 止盈条件
                    if position * spread_change > self.take_profit:
                        pair_info['positions'].append({
                            'date': date,
                            'position': 0,
                            's1_price': s1_price,
                            's2_price': s2_price,
                            'z_score': current_z_score,
                            'reason': '止盈'
                        })
                        pair_info['position'] = 0
                        pair_info['shares'] = (0, 0)

                    # 止损条件
                    elif abs(current_z_score) > self.stop_loss:
                        pair_info['positions'].append({
                            'date': date,
                            'position': 0,
                            's1_price': s1_price,
                            's2_price': s2_price,
                            'z_score': current_z_score,
                            'reason': '止损'
                        })
                        pair_info['position'] = 0
                        pair_info['shares'] = (0, 0)

                    # 回归条件
                    elif (position > 0 and current_z_score < 0) or (position < 0 and current_z_score > 0):
                        pair_info['positions'].append({
                            'date': date,
                            'position': 0,
                            's1_price': s1_price,
                            's2_price': s2_price,
                            'z_score': current_z_score,
                            'reason': '回归'
                        })
                        pair_info['position'] = 0
                        pair_info['shares'] = (0, 0)

            # 更新组合价值
            current_capital = 0
            for pair_info in self.selected_pairs:
                pair = pair_info['pair']
                s1, s2 = pair
                s1_price = self.stock_data[s1].loc[date, '收盘价']
                s2_price = self.stock_data[s2].loc[date, '收盘价']
                shares_s1, shares_s2 = pair_info['shares']
                current_capital += shares_s1 * s1_price + shares_s2 * s2_price

            self.portfolio_value.append((date, current_capital))

        # 计算回测绩效指标
        return self.calculate_performance()

    def calculate_performance(self):
        """计算回测绩效指标"""
        if not self.portfolio_value:
            return {}

        # 计算每日收益率
        dates, values = zip(*self.portfolio_value)
        daily_returns = []
        for i in range(1, len(values)):
            ret = (values[i] - values[i - 1]) / values[i - 1]
            daily_returns.append(ret)

        # 计算累积收益率
        total_return = (values[-1] - values[0]) / values[0]

        # 计算年化收益率
        days = (dates[-1] - dates[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1

        # 计算夏普比率
        if len(daily_returns) > 0:
            daily_sharpe = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
            annual_sharpe = daily_sharpe * np.sqrt(252)
        else:
            annual_sharpe = 0

        # 计算最大回撤
        max_drawdown = 0
        peak = values[0]
        for value in values:
            if value > peak:
                peak = value
            drawdown = (value - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown

        return {
            '总收益率': total_return,
            '年化收益率': annual_return,
            '夏普比率': annual_sharpe,
            '最大回撤': max_drawdown
        }


# 主程序
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='配对交易策略回测')
    parser.add_argument('--start_date', type=str, default='2024-01-01',
                        help='回测开始日期，格式: YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default='2025-04-30',
                        help='回测结束日期，格式: YYYY-MM-DD')
    parser.add_argument('--output_dir', type=str, default='../fig/results',
                        help='结果输出目录')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载行业分类数据
    industry_df = pd.read_csv("zz500行业分类.csv", encoding='utf-8')
    industry_df = industry_df.rename(columns={
        "证券代码": "stock_code",
        "证券简称": "stock_name",
        "上市日期": "list_date",
        "所属行业": "industry"
    })

    # 2. 加载股票价格数据
    stock_folder = "../data_trading"
    stock_data = {}

    for filename in os.listdir(stock_folder):
        if filename.endswith(".csv"):
            stock_code = filename.replace(".csv", "")
            file_path = os.path.join(stock_folder, filename)
            df = pd.read_csv(file_path, encoding='utf-8')
            df['时间'] = pd.to_datetime(df['时间'])
            df.set_index('时间', inplace=True)

            # 数据清洗：移除包含NaN的行
            df = df.dropna(subset=['收盘价', '成交额.元.'])

            # 检查价格是否为零或负值
            if (df['收盘价'] <= 0).any():
                print(f"警告：股票 {stock_code} 包含非正价格，跳过此股票")
                continue

            stock_data[stock_code] = df[['收盘价', '成交额.元.']]

    # 3. 初始化策略类
    # 可以调整策略参数
    strategy = PairTradingStrategy(
        stock_data,
        industry_df,
        formation_period=252,  # 形成期长度
        z_threshold=1.0,  # Z得分阈值
        stop_loss=3.0,  # 止损阈值
        take_profit=0.2,  # 止盈阈值（10%收益）
        initial_capital=1000000  # 初始资金
    )

    # 4. 指定股票对
    # 这里可以修改为你想要的股票对
    # 格式：[(股票1代码, 股票2代码), (股票3代码, 股票4代码), ...]
    specified_pairs = [
        # ('603596.SH', '603129.SH'),  # 示例股票对1
        ('600295.SH', '000825.SZ')  # 示例股票对2


    ]

    # 5. 处理指定的股票对（取消协整检验）
    print("正在处理指定的股票对...")
    for pair in specified_pairs:
        s1, s2 = pair

        # 检查股票是否存在于数据中
        if s1 not in stock_data:
            print(f"警告：股票 {s1} 不存在于数据中，跳过此股票对")
            continue
        if s2 not in stock_data:
            print(f"警告：股票 {s2} 不存在于数据中，跳过此股票对")
            continue

        # 获取形成期数据
        if len(stock_data[s1]) < strategy.formation_period or len(stock_data[s2]) < strategy.formation_period:
            print(f"警告：股票 {s1} 或 {s2} 的数据不足，跳过此股票对")
            continue

        s1_data = stock_data[s1].iloc[:strategy.formation_period]
        s2_data = stock_data[s2].iloc[:strategy.formation_period]

        # 确保两个股票的日期完全对齐
        common_dates = s1_data.index.intersection(s2_data.index)
        if len(common_dates) < strategy.formation_period * 0.9:  # 要求至少90%的日期匹配
            print(f"警告：股票 {s1} 和 {s2} 的日期匹配度不足，跳过此股票对")
            continue

        s1_data = s1_data.loc[common_dates]
        s2_data = s2_data.loc[common_dates]

        # 计算价差
        spread = s1_data['收盘价'] - s2_data['收盘价']

        # 检查价差序列是否有效
        if spread.std() < 1e-10:  # 价差几乎恒定
            print(f"警告：股票对 {pair} 的价差几乎恒定，跳过此股票对")
            continue

        if np.isnan(spread).any() or np.isinf(spread).any():
            print(f"警告：股票对 {pair} 的价差包含NaN或inf，跳过此股票对")
            continue

        # 计算价差统计特性
        mean = spread.mean()
        std = spread.std()

        # 计算半衰期（增加异常处理）
        try:
            half_life = strategy._calculate_half_life(spread)
            if np.isnan(half_life) or np.isinf(half_life) or half_life <= 0:
                raise ValueError("无效的半衰期值")
        except Exception as e:
            print(f"警告：计算股票对 {pair} 的半衰期时出错: {str(e)}，使用默认值20天")
            half_life = 20  # 默认半衰期值

        # 计算日均成交额
        volume_s1 = s1_data['成交额.元.']
        volume_s2 = s2_data['成交额.元.']
        avg_volume_s1 = volume_s1.mean()
        avg_volume_s2 = volume_s2.mean()
        total_avg_volume = avg_volume_s1 + avg_volume_s2

        strategy.selected_pairs.append({
            'pair': pair,
            'spread_mean': mean,
            'spread_std': std,
            'half_life': half_life,
            'total_avg_volume': total_avg_volume,
            'position': 0,  # 当前仓位
            'entry_price': 0,  # 入场价差
            'positions': []  # 交易信号记录
        })

        print(f"已添加股票对 {pair}，价差标准差: {std:.4f}，半衰期: {half_life:.2f}天")

    # 6. 检查是否有有效的股票对
    if not strategy.selected_pairs:
        print("没有找到有效的股票对进行回测，请检查指定的股票对和数据")
        return

    # 7. 输出选择的股票对
    print("\n最终选择的股票对：")
    for i, p in enumerate(strategy.selected_pairs):
        print(
            f"{i + 1}. 股票对: {p['pair']}, 价差标准差: {p['spread_std']:.4f}, 日均成交额: {p['total_avg_volume']:.2f}元")

    # 8. 回测策略
    print(f"\n开始回测，日期范围: {args.start_date} 至 {args.end_date}...")
    start_date = pd.to_datetime(args.start_date)
    end_date = pd.to_datetime(args.end_date)
    performance = strategy.backtest(start_date, end_date)

    # 9. 输出回测结果
    print("\n策略表现：")
    for key, value in performance.items():
        if key == '夏普比率' or key == '最大回撤':
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value:.2%}")

    # 10. 为每个股票对绘制价格走势图及交易信号
    # 设置中文字体
    # plt.rcParams["font.family"] = ["Heiti TC", "SimHei", "WenQuanYi Micro Hei"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    # 自定义格式化函数，禁用科学记数法
    def format_currency(x, pos):
        return f'{int(x):,}'

    # 绘制每个股票对的价格走势图
    for i, pair_info in enumerate(strategy.selected_pairs):
        pair = pair_info['pair']
        s1, s2 = pair

        # 获取回测期间的价格数据
        dates = []
        s1_prices = []
        s2_prices = []

        for date in pd.date_range(start=start_date, end=end_date):
            if date in stock_data[s1].index and date in stock_data[s2].index:
                price1 = stock_data[s1].loc[date, '收盘价']
                price2 = stock_data[s2].loc[date, '收盘价']

                dates.append(date)
                s1_prices.append(price1)
                s2_prices.append(price2)

        # 获取交易信号
        signals = pair_info.get('positions', [])
        buy_dates_s1 = []
        buy_prices_s1 = []
        sell_dates_s1 = []
        sell_prices_s1 = []

        buy_dates_s2 = []
        buy_prices_s2 = []
        sell_dates_s2 = []
        sell_prices_s2 = []

        for signal in signals:
            signal_date = signal['date']
            position = signal['position']

            if signal_date in dates:
                date_idx = dates.index(signal_date)
                s1_price = s1_prices[date_idx]
                s2_price = s2_prices[date_idx]

                # 股票1的信号
                if position == 1:  # 做多股票1
                    buy_dates_s1.append(signal_date)
                    buy_prices_s1.append(s1_price)
                elif position == -1:  # 做空股票1
                    sell_dates_s1.append(signal_date)
                    sell_prices_s1.append(s1_price)

                # 股票2的信号（与股票1相反）
                if position == 1:  # 做空股票2
                    sell_dates_s2.append(signal_date)
                    sell_prices_s2.append(s2_price)
                elif position == -1:  # 做多股票2
                    buy_dates_s2.append(signal_date)
                    buy_prices_s2.append(s2_price)

                # 创建图表
            fig, ax1 = plt.subplots(figsize=(16, 8))

            # 绘制股票1（s1）价格（左侧Y轴）
            ax1.set_xlabel('日期')
            ax1.set_ylabel(f'{s1} 价格 (元)', color='black')
            line_s1 = ax1.plot(dates, s1_prices, color='blue', linewidth=1.5, label=s1)
            ax1.tick_params(axis='y', labelcolor='black')

            # 绘制股票2（s2）价格（右侧Y轴）
            ax2 = ax1.twinx()
            ax2.set_ylabel(f'{s2} 价格 (元)', color='black')
            line_s2 = ax2.plot(dates, s2_prices, color='orange', linewidth=1.5, label=s2)
            ax2.tick_params(axis='y', labelcolor='black')

            # **标记交易信号并区分股票代码**
            # 股票1（s1）的做多信号（绿色向上箭头，标签包含s1）
            buy_s1 = ax1.scatter(
                buy_dates_s1, buy_prices_s1,
                marker='^', color='green', s=50,
                edgecolors='black', linewidths=0.5,
                label=f'{s1} 做多'
            )
            # 股票1（s1）的做空信号（红色向下箭头，标签包含s1）
            sell_s1 = ax1.scatter(
                sell_dates_s1, sell_prices_s1,
                marker='v', color='red', s=50,
                edgecolors='black', linewidths=0.5,
                label=f'{s1} 做空'
            )
            # 股票2（s2）的做多信号（紫色向上箭头，标签包含s2）
            buy_s2 = ax2.scatter(
                buy_dates_s2, buy_prices_s2,
                marker='^', color='purple', s=50,
                edgecolors='black', linewidths=0.5,
                label=f'{s2} 做多'
            )
            # 股票2（s2）的做空信号（棕色向下箭头，标签包含s2）
            sell_s2 = ax2.scatter(
                sell_dates_s2, sell_prices_s2,
                marker='v', color='brown', s=50,
                edgecolors='black', linewidths=0.5,
                label=f'{s2} 做空'
            )

            # **手动组合图例句柄和标签（顺序与绘制顺序一致）**
            handles = [line_s1[0], line_s2[0], buy_s1, sell_s1, buy_s2, sell_s2]
            labels = [s1, s2, f'{s1} 做多', f'{s1} 做空', f'{s2} 做多', f'{s2} 做空']

            # **添加图例（分两行显示，位置在图表外上方）**
            ax1.legend(
                handles, labels,
                loc='upper right',  # 图例位于右下角
                frameon=True,
                framealpha=0.9,
                title='图例',  # 可选：添加图例标题
                title_fontsize=10,
                # bbox_to_anchor=(1, 1),  # 调整锚点避免超出图表边界
                # ncol=2  # 若条目过多，可设置列数
            )

            # 设置x轴日期格式和网格线（省略重复逻辑）
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()
            ax1.grid(True, linestyle='--', alpha=0.7)

            # 添加标题（省略重复逻辑）
            plt.title(f'股票对 {pair} 价格走势图及交易信号 ({args.start_date}至{args.end_date})')

        # 保存价格走势图
        pair_img_path = os.path.join(args.output_dir, f'prices_{s1}_{s2}_signals_{args.start_date}_{args.end_date}.png')
        plt.tight_layout()
        plt.savefig(pair_img_path, dpi=300, bbox_inches='tight')
        print(f"股票对 {pair} 的价格走势图已保存至: {pair_img_path}")
        plt.close()  # 关闭当前图表，避免内存占用

    # 11. 保存回测结果到CSV
    results = {
        '日期': [d.strftime('%Y-%m-%d') for d, _ in strategy.portfolio_value],
        '组合价值': [v for _, v in strategy.portfolio_value]
    }

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(args.output_dir, f'backtest_results_{args.start_date}_{args.end_date}.csv')
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"回测结果已保存至: {results_csv_path}")


if __name__ == "__main__":
    main()