from Model import *
import pickle
import torch
import argparse
import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score
import csv

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



def evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=None, top_k=5):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []  # 初始化一个空列表
    trues = []
    top_k_stocks_all_days = []  # 用于存储每一天的推荐股票

    for i in seq:
        output = model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], relation_static=relation_static)
        output = output.detach().cpu().numpy()  # 转换为 NumPy 数组
        preds.append(np.exp(output))  # 将计算出的值加入到列表中
        trues.append(y_eval[i][:73].cpu().numpy())  # 保存真实值

        # 选择当前天的 top_k 股票
        top_k_indices = np.argsort(-output[:, 1])[:top_k]  # 选择预测值最高的前 top_k 个股票
        top_k_stocks_all_days.append(top_k_indices)  # 存储每一天的推荐股票

    # 将预测和真实值转换为 NumPy 数组
    preds = np.array(preds).reshape(-1, preds[0].shape[-1])
    trues = np.array(trues).reshape(-1, trues[0].shape[-1])

    # 计算 AUC、准确率、MCC 和 DA
    auc = roc_auc_score(trues[:, 0], preds[:, 1])
    acc = accuracy_score(trues[:, 0], np.argmax(preds, axis=1))
    mcc = matthews_corrcoef(trues[:, 0], np.argmax(preds, axis=1))
    da = np.mean(np.sign(trues[:, 0] - 0.5) == np.sign(preds[:, 1] - 0.5))

    # 写入 CSV 文件
    with open('data/top_k_stocks.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', f'Top {top_k} recommended stocks'])  # 写入表头

        for day, top_k_stocks in enumerate(top_k_stocks_all_days):
            writer.writerow([f"Day {day+1}", top_k_stocks])  # 写入每一天的推荐股票

    print(f"Top k stocks for each day:")
    for day, top_k_stocks in enumerate(top_k_stocks_all_days):
        print(f"Day {day+1}: Top {top_k} recommended stocks: {top_k_stocks}")

    return acc, auc, mcc, da, top_k_stocks_all_days





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

    # Evaluate and get top 5 recommended stocks
    test_acc, test_auc, test_mcc, test_da, top_k_stocks = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static, top_k=15)

    print("测试集结果:")
    print(f"准确率: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"MCC: {test_mcc:.4f}")
    print(f"DA: {test_da:.4f}")
    print(f"Top k stocks for each day: {top_k_stocks}")
