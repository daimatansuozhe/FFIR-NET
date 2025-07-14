from Model import *
from utils import *
import pickle
import torch
import argparse
import numpy as np
from sklearn.metrics import matthews_corrcoef

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='GPU to use')
parser.add_argument('--relation', type=str, default='None', help='all, competitor, customer, industry, strategic, supply')

def load_dataset(DEVICE):
    with open('./data/x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('./data/y_.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./data/x_textual.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle)
    with open('close_datasp500.pkl', 'rb') as handle:  # 新增部分：加载收盘价数据
        stock_close_prices = pickle.load(handle)  # 假设它的维度是 [天数, 股票数, FFIR-NET]

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, dtype=torch.float64).to(DEVICE)
    x_sentiment = torch.tensor(stock_sentiments, dtype=torch.float64).to(DEVICE)
    stock_close_prices = torch.tensor(stock_close_prices, dtype=torch.float64).to(DEVICE)  # 新增：将收盘价数据转为tensor

    if args.relation != "None":
        with open('./data/relations/' + args.relation + '_relation.pkl', 'rb') as handle:
            relation_static = pickle.load(handle)
        if isinstance(relation_static, dict):
            for key in relation_static:
                relation_static[key] = torch.tensor(relation_static[key], dtype=torch.float64).to(DEVICE)
        else:
            relation_static = torch.tensor(relation_static, dtype=torch.float64).to(DEVICE)
    else:
        relation_static = None

    y = torch.tensor(y_load, dtype=torch.float64).to(DEVICE)
    y = (y > 0).to(torch.long).to(DEVICE)

    return x, y, x_sentiment, relation_static, stock_close_prices  # 返回收盘价数据

# 模拟投资收益，结合模型推荐和实际收盘价
def simulate_investment(budget, top_k_stocks_all_days, stock_close_prices, transaction_cost=0.0003):
    daily_funds = [budget]

    for day in range(1, len(top_k_stocks_all_days)):
        top_stocks = top_k_stocks_all_days[day - 1]  # 前一天的 top_k 股票
        today_prices = stock_close_prices[day].squeeze().cpu().numpy()  # 获取当天的收盘价
        yesterday_prices = stock_close_prices[day - 1].squeeze().cpu().numpy()  # 获取前一天的收盘价

        # 确保推荐的股票有有效的价格
        valid_stocks = [stock for stock in top_stocks if stock < len(today_prices) and stock < len(yesterday_prices) and today_prices[stock] > -1 and yesterday_prices[stock] > -1]

        if len(valid_stocks) == 0:
            daily_funds.append(daily_funds[-1])  # 没有有效股票，资金保持不变
            print(f"No valid stocks for Day {day + 1}, keeping previous day's funds.")
            continue

        # 每只股票的平均投资
        investment_per_stock = daily_funds[-1] / len(valid_stocks)  
        new_budget = 0

        for stock in valid_stocks:
            # 计算增长率
            growth_rate = (today_prices[stock] - yesterday_prices[stock]) / yesterday_prices[stock]
            # 更新预算，扣除交易成本
            new_budget += investment_per_stock * (1 + growth_rate) * (1 - transaction_cost)

        # 记录当天的总资金
        daily_funds.append(new_budget)

    return daily_funds

# 评估模型，选择 top_k 股票
def evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=None, top_k=198):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    top_k_stocks_all_days = []  # 每天的前 top_k 股票

    for i in seq:
        output = model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], relation_static=relation_static)
        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i].cpu().numpy())

        # 选出 top_k 的股票
        top_k_indices = np.argsort(-output[:, 1])[:top_k]
        top_k_stocks_all_days.append(top_k_indices)

    # print(f"x_test shape: {x_eval.shape}")
    # print(f"x_sentiment_test shape: {x_sentiment_eval.shape}")
    # print(f"y_test shape: {y_eval.shape}")
    # print(f"Sequence range: {seq}")
    acc, auc = metrics(trues, preds)
    preds = np.array(preds).squeeze()
    trues = np.array(trues).squeeze()
    preds = np.argmax(preds, axis=1)
    trues = np.argmax(trues, axis=1)
    preds = np.argmax(preds, axis=1)
    mcc = matthews_corrcoef(trues, preds)
    da = np.mean(np.sign(preds) == np.sign(trues))

    return acc, auc, mcc, da, top_k_stocks_all_days

if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")

    print("loading dataset")
    x, y, x_sentiment, relation_static, stock_close_prices = load_dataset(DEVICE)  # 加载收盘价数据

    rnn_length = 20
    # x_test = x[-75 - rnn_length:-25]
    # y_test = y[-75 - rnn_length:-25]
    # x_sentiment_test = x_sentiment[-75 - rnn_length:-25]
    # stock_close_prices_test = stock_close_prices[-75 - rnn_length:-25]  # 只取测试集部分的收盘价

    x_test = x[ 515:535+ rnn_length ]
    y_test = y[ 515:535+ rnn_length ]
    x_sentiment_test = x_sentiment[ 515:535 + rnn_length ]
    stock_close_prices_test = stock_close_prices[ 515:535+ rnn_length ]
    # print(f"x_test shape: {x_test.shape}")
    # print(f"y_test shape: {y_test.shape}")

    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)

    model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS,
                   d_hidden=D_MARKET, hidn_rnn=360, heads_att=6,
                   hidn_att=60, dropout=0.2, t_mix=1,
                   infer=1, relation_static=relation_static)

    model = model.to(DEVICE)
    best_model_file = ("./SavedModels/tiaocan/epoch37_eval_auc0.5193118921041805_acc0.5165945165945166_mcc-0.014185199879352394_da0.5_test_auc0.5907663471909528_acc0.5696969696969697_mcc-0.023284921516163435_da0.4857142857142857")
    model.load_state_dict(torch.load(best_model_file))

    # Evaluate and get top 15 recommended stocks
    test_acc, test_auc, test_mcc, test_da, top_k_stocks = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static, top_k=198)

    # Simulate investment using the actual closing prices
    initial_budget = 10000  # 初始资金
    daily_funds = simulate_investment(initial_budget, top_k_stocks, stock_close_prices_test)  # 传入收盘价

    # 输出每一天的总资金
    for day, funds in enumerate(daily_funds):
        # print(f"Day {day+FFIR-NET}: Total funds: {funds:.2f}元")
        print(f"{funds:.2f}")
