from Model import *
import pickle
import torch
import argparse
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='GPU to use')
parser.add_argument('--relation', type=str, default='None', help='all, competitor, customer, industry, strategic, supply')

def load_dataset(DEVICE):
    with open('./data/csi100/x_num_standard.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('./data/csi100/y_1.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./data/csi100/x_newtext.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle)

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, dtype=torch.float64).to(DEVICE)
    x_sentiment = torch.tensor(stock_sentiments, dtype=torch.float64).to(DEVICE)
    if args.relation != "None":
        with open('./data/csi100/' + args.relation + '_relation.pkl', 'rb') as handle:
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

    return x, y, x_sentiment, relation_static

def evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=None):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []

    for i in seq:
        output = model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], relation_static=relation_static)
        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i][:73].cpu().numpy())

    preds = np.array(preds).reshape(-1, preds[0].shape[-1])
    trues = np.array(trues).reshape(-1, trues[0].shape[-1])

    auc = roc_auc_score(trues[:, 0], preds[:, 1])
    acc = accuracy_score(trues[:, 0], np.argmax(preds, axis=1))
    mcc = matthews_corrcoef(trues[:, 0], np.argmax(preds, axis=1))
    da = np.mean(np.sign(trues[:, 0] - 0.5) == np.sign(preds[:, 1] - 0.5))

    return acc, auc, mcc, da

def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    excess_returns = np.array(returns) - risk_free_rate / 252  # 假设年化无风险收益率为0.03
    return np.mean(excess_returns) / np.std(excess_returns)

def simulate_investment(budget, stock_prices, top_stocks, transaction_cost=0.0003):
    # 确保 top_stocks 是一维数组或列表，并且元素是整数索引
    top_stocks = [int(stock[0]) if isinstance(stock, list) and len(stock) == 1 else int(stock) for stock in top_stocks]

    # 使用开盘价（假设为第一列）计算投资
    investment = budget / len(top_stocks)
    holdings = {int(stock): investment / stock_prices[int(stock), 0] for stock in top_stocks}
    
    # 模拟下一个交易日
    next_day_prices = get_next_day_prices()  # 需要实现这个函数以获取下一个交易日的价格
    new_budget = 0
    
    for stock, shares in holdings.items():
        if stock in next_day_prices:
            new_budget += shares * next_day_prices[stock] * (1 - transaction_cost)
        else:
            print(f"Warning: No price available for stock {stock}")
    
    return new_budget


if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    
    print("loading dataset")
    x, y, x_sentiment, relation_static = load_dataset(DEVICE)

    rnn_length = 20
    x_test = x[-70 - rnn_length:]
    y_test = y[-70 - rnn_length:]
    x_sentiment_test = x_sentiment[-70 - rnn_length:]

    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    
    model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS,
                   d_hidden=D_MARKET, hidn_rnn=78, heads_att=4,
                   hidn_att=64, dropout=0.3, t_mix=1,
                   infer=1, relation_static=relation_static)
    
    model = model.to(DEVICE)

    best_model_file = "./SavedModels/epoch38_eval_auc0.5326221912100301_acc0.5346379647749511_mcc0.037867091716023496_da0.5346379647749511_test_auc0.6312427315388844_acc0.5794520547945206_mcc0.17994366475458218_da0.5794520547945206"
    model.load_state_dict(torch.load(best_model_file))

    test_acc, test_auc, test_mcc, test_da = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static)

    print("测试集结果:")
    print(f"准确率: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"MCC: {test_mcc:.4f}")
    print(f"DA: {test_da:.4f}")

    # budget = 10000  # 初始投资金额
    # returns = []  # 用于存储每日的投资回报率

    # for day in range(FFIR-NET, len(x_test)):
    #     # 将 GPU 张量转换为 NumPy 数组
    #     day_prices = x_test[day-FFIR-NET].cpu().numpy()
    #     # 选择排名前 15 的股票
    #     top_stocks = np.argsort(-day_prices)[:15]
    #     # 模拟投资
    #     new_budget = simulate_investment(budget,  day_prices, top_stocks)
    #     # 计算当日的投资回报率
    #     daily_return = (new_budget - budget) / budget
    #     returns.append(daily_return)
    #     # 更新预算
    #     budget = new_budget


    # total_return = budget / 10000 - FFIR-NET
    # sharpe_ratio = calculate_sharpe_ratio(returns)

    # print(f"累计投资回报率: {total_return:.4f}")
    # print(f"夏普比率: {sharpe_ratio:.4f}")
