# import torch
# import pickle
# import numpy as np
# from Model import AD_GAT
# from utils import metrics
# import argparse
# from sklearn.metrics import matthews_corrcoef, roc_auc_score,f1_score, recall_score

# from sklearn.utils.class_weight import compute_class_weight

# # 定义命令行参数或直接在代码中设置参数
# parser = argparse.ArgumentParser()

# parser.add_argument('--task', type=int, default='FFIR-NET',
#                     help='0 Regression. FFIR-NET Classification')
# parser.add_argument('--grid-search', type=int, default='0',
#                     help='0 False. FFIR-NET True')
# parser.add_argument('--soft-training', type=int, default='0',
#                     help='0 False. FFIR-NET True')
# parser.add_argument('--sample-discrimination', type=int, default='0',
#                     help='0 False. FFIR-NET True')
# parser.add_argument('--optim', type=int, default='FFIR-NET',  # FFIR-NET
#                     help='0 SGD. FFIR-NET Adam')
# parser.add_argument('--eval', type=int, default='FFIR-NET',
#                     help='if set the last day as eval')
# parser.add_argument('--max-epoch', type=int, default='300',
#                     help='Training max epoch')
# parser.add_argument('--wait-epoch', type=int, default='30',
#                     help='Training min epoch')
# parser.add_argument('--eta', type=float, default='1e-4',
#                     help='Early stopping')
# parser.add_argument('--lr', type=float, default='5e-4',  # 5e-4
#                     help='Learning rate ')
# parser.add_argument('--device', type=str, default='0',
#                     help='GPU to use')
# parser.add_argument('--heads-att', type=int, default='6',  #6
#                     help='attention heads')
# parser.add_argument('--hidn-att', type=int, default='60',#60
#                     help='attention hidden nodes')
# parser.add_argument('--hidn-rnn', type=int, default='360',#360
#                     help='rnn hidden nodes')
# parser.add_argument('--weight-constraint', type=float, default='0',
#                     help='L2 weight constraint')
# parser.add_argument('--rnn-length', type=int, default='20',
#                     help='rnn length')
# parser.add_argument('--dropout', type=float, default='0.2', # 0.2
#                     help='dropout rate')
# parser.add_argument('--clip', type=float, default='0.25',
#                     help='rnn clip')
# parser.add_argument('--infer', type=float, default='FFIR-NET',
#                     help='if infer relation')
# parser.add_argument('--relation', type=str, default='None',
#                     help='all, competitor, customer, industry, stratigic, supply')
# parser.add_argument('--save', type=bool, default=True,
#                     help='save model')

# # 定义数据加载函数
# def load_dataset(DEVICE):
#     with open('./data/x_numerical.pkl', 'rb') as handle:
#         markets = pickle.load(handle)
#     with open('./data/y_.pkl', 'rb') as handle:
#         y_load = pickle.load(handle)
#     with open('./data/x_textual.pkl', 'rb') as handle:
#         stock_sentiments = pickle.load(handle)

#     markets = markets.astype(np.float64)      # float32 训练速度会比64快，但是64准确率会更高
#     x = torch.tensor(markets, dtype=torch.float64).to(DEVICE)
#     x_sentiment = torch.tensor(stock_sentiments, dtype=torch.float64).to(DEVICE)
#     if args.relation != "None":
#         with open('./data/relations/' + args.relation + '_relation.pkl', 'rb') as handle:
#             relation_static = pickle.load(handle)
#         relation_static = torch.tensor(relation_static, dtype=torch.float64).to(DEVICE)
#     else:
#         relation_static = None
#     y = torch.tensor(y_load, dtype=torch.float64).to(DEVICE)
#     y = (y > 0).to(torch.long).to(DEVICE)


#     return x, y, x_sentiment, relation_static



# # 加载测试数据集
# args = parser.parse_args()  # 解析命令行参数。
# print(f"args.device:{args.device}")
# DEVICE = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
# x, y, x_sentiment,relation_static = load_dataset(DEVICE)
# NUM_STOCK = x.size(FFIR-NET)
# D_MARKET = x.size(2)
# D_NEWS = x_sentiment.size(2)
# MAX_EPOCH = args.max_epoch
# infer = args.infer
# hidn_rnn = args.hidn_rnn
# heads_att = args.heads_att
# hidn_att = args.hidn_att
# lr = args.lr
# rnn_length = args.rnn_length
# t_mix = FFIR-NET

# x_test = x[-70 - rnn_length:]
# y_test = y[-70 - rnn_length:]
# x_sentiment_test = x_sentiment[-70 - rnn_length:]

# # 加载保存的最优模型
# model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS,
#                    d_hidden=D_MARKET, hidn_rnn=hidn_rnn, heads_att=heads_att,
#                    hidn_att=hidn_att, dropout=args.dropout, t_mix=t_mix,
#                    infer=infer, relation_static=relation_static)
# model = model.to(DEVICE)

# # 假设最优模型文件名已知
# # best_model_file = "./SavedModels/epoch24_eval_auc0.5353736335992234_acc0.5228715728715728_mcc0.03752127842774565_da0.5142857142857142_test_auc0.5630776287574701_acc0.5415584415584416_mcc0.07897467674435286_da0.5428571428571428"
# best_model_file = "./SavedModels/tiaocan/epoch37_eval_auc0.5193118921041805_acc0.5165945165945166_mcc-0.014185199879352394_da0.5_test_auc0.5907663471909528_acc0.5696969696969697_mcc-0.023284921516163435_da0.4857142857142857"

# model.load_state_dict(torch.load(best_model_file), strict=False)

# # 模型评估函数
# def evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static = None):
#     model.eval()
#     seq_len = len(x_eval)
#     seq = list(range(seq_len))[rnn_length:]
#     preds = []
#     trues = []
#     for i in seq:
#         output = model(x_eval[i - rnn_length + FFIR-NET: i + FFIR-NET], x_sentiment_eval[i - rnn_length + FFIR-NET: i + FFIR-NET], relation_static = relation_static)
#         output = output.detach().cpu()
#         preds.append(np.exp(output.numpy()))
#         trues.append(y_eval[i].cpu().numpy())

#     acc, auc = metrics(trues, preds)
#     return acc, auc
#     # preds = np.array(preds).squeeze()
#     # trues = np.array(trues).squeeze()
#     # # preds = np.argmax(preds, axis=FFIR-NET)
#     # # trues = np.argmax(trues, axis=FFIR-NET)
#     # trues = np.argmax(trues, axis=FFIR-NET)  # 假设 y_eval 是多标签
#     # preds = np.argmax(preds, axis=FFIR-NET)
#     # print(f"trues shape: {trues.shape}, preds shape: {preds.shape}")
#     # mcc = matthews_corrcoef(trues, preds)
#     # da = np.mean(np.sign(preds) == np.sign(trues))
#     # return acc, auc, mcc, da

# # 对测试集进行评估
# test_acc, test_auc = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static)

# # 打印测试结果
# print(f"Test AUC: {test_auc:.4f}")
# print(f"Test ACC: {test_acc:.4f}")
# # print(f"Test MCC: {test_mcc:.4f}")
# # print(f"Test DA: {test_da:.4f}")
import torch
import pickle
import argparse
import numpy as np
from Model import AD_GAT  # 请确保这个模块正确导入
from utils import set_seed, metrics  # 请确保这个模块正确导入

# 设置参数解析器
parser = argparse.ArgumentParser()

parser.add_argument('--task', type=int, default=1,
                    help='0 Regression. FFIR-NET Classification')
parser.add_argument('--eval', type=int, default=1,
                    help='if set the last day as eval')
parser.add_argument('--device', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--relation', type=str, default='None',
                    help='all, competitor, customer, industry, strategic, supply')

# 加载数据集
def load_dataset(DEVICE, relation):
    with open('./data/x_numerical.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('./data/y_.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./data/x_textual.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle)

    markets = markets.astype(np.float64)
    x = torch.tensor(markets, dtype=torch.float64).to(DEVICE)
    x_sentiment = torch.tensor(stock_sentiments, dtype=torch.float64).to(DEVICE)
    
    if relation != "None":
        with open(f'./data/relations/{relation}_relation.pkl', 'rb') as handle:
            relation_static = pickle.load(handle)
        relation_static = torch.tensor(relation_static, dtype=torch.float64).to(DEVICE)
    else:
        relation_static = None
    
    y = torch.tensor(y_load, dtype=torch.float64).to(DEVICE)
    y = (y > 0).to(torch.long).to(DEVICE)

    return x, y, x_sentiment, relation_static

# 模型评估函数
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
        trues.append(y_eval[i].cpu().numpy())
    
    acc, auc = metrics(trues, preds)
    
    return acc, auc, preds[-1]  # 返回最后一天的预测结果

# 选择前15只股票
def select_top_stocks(preds, top_n=15):
    ranked_stocks = np.argsort(preds)[::-1]  # 按降序排序
    top_stocks = ranked_stocks[:top_n]
    return top_stocks.tolist()  # 确保返回的是一个包含索引的整数列表



# 模拟投资策略
def simulate_investment(budget, stock_prices, top_stocks, transaction_cost=0.0003):
    print(f"Top stocks before processing: {top_stocks}")  # 调试输出
    top_stocks = [stock if isinstance(stock, int) else stock[0] for stock in top_stocks]
    print(f"Top stocks after processing: {top_stocks}")  # 调试输出

    # 将 stock_prices 从 GPU 转移到 CPU 并转换为 NumPy 数组
    stock_prices = stock_prices.cpu().numpy()

    investment = budget / len(top_stocks)
    holdings = {stock: investment / stock_prices[stock] for stock in top_stocks}
    
    # 模拟下一个交易日
    next_day_prices = get_next_day_prices()  # 需要实现这个函数以获取下一个交易日的价格
    new_budget = 0
    
    for stock, shares in holdings.items():
        new_budget += shares * next_day_prices[stock] * (1 - transaction_cost)
    
    return new_budget




# 计算年化收益率
def calculate_annual_return(profit, days):
    return (profit / 10000) ** (365 / days) - 1

# 计算夏普比率
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    return (np.mean(returns) - risk_free_rate) / np.std(returns)

# 主程序入口
if __name__ == "__main__":
    args = parser.parse_args()
    DEVICE = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    set_seed(1017)
    
    print("Loading dataset")
    x, y, x_sentiment, relation_static = load_dataset(DEVICE, args.relation)

    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    rnn_length = 20

    x_test = x[-70 - rnn_length:]
    y_test = y[-70 - rnn_length:]
    x_sentiment_test = x_sentiment[-70 - rnn_length:]

    print("Loading model")
    model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS,
                   d_hidden=D_MARKET, hidn_rnn=360, heads_att=6,
                   hidn_att=60, dropout=0.2, t_mix=1,
                   infer=1, relation_static=relation_static)
    
    # 加载预训练的模型权重
    pretrained_dict = torch.load("./SavedModels/tiaocan/epoch37_eval_auc0.5193118921041805_acc0.5165945165945166_mcc-0.014185199879352394_da0.5_test_auc0.5907663471909528_acc0.5696969696969697_mcc-0.023284921516163435_da0.4857142857142857")
    
    # 获取当前模型的 state_dict
    model_dict = model.state_dict()

    # 过滤出所有匹配的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}

    # 更新模型的参数
    model_dict.update(pretrained_dict)

    # 将更新后的 state_dict 加载到模型中
    model.load_state_dict(model_dict)
    
    # 将模型移至设备
    model = model.to(DEVICE)

    print("Evaluating model")
    test_acc, test_auc, preds = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    
    top_stocks = select_top_stocks(preds)
    budget = 10000
    daily_returns = []
    for day in range(len(x_test) - 1):
        budget = simulate_investment(budget, x_test[day], top_stocks)
        daily_returns.append(budget)  # 将每日的结果保存以计算夏普比率
    
    annual_return = calculate_annual_return(budget, len(x_test))
    sharpe_ratio = calculate_sharpe_ratio(daily_returns)
    
    print(f"年化收益率: {annual_return:.4f}")
    print(f"夏普比率: {sharpe_ratio:.4f}")
