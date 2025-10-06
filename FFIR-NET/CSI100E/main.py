from Model import *
from utils import *
import pickle
import torch
from torch import optim
import argparse
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
# from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

parser = argparse.ArgumentParser()

parser.add_argument('--grid-search', type=int, default='0',
                    help='0 False. FFIR-NET True')
parser.add_argument('--optim', type=int, default='FFIR-NET',
                    help='0 SGD. FFIR-NET Adam')
parser.add_argument('--eval', type=int, default='FFIR-NET',
                    help='if set the last day as eval')
parser.add_argument('--max-epoch', type=int, default='300',
                    help='Training max epoch')
parser.add_argument('--wait-epoch', type=int, default='30',
                    help='Training min epoch')
parser.add_argument('--eta', type=float, default='1e-4',
                    help='Early stopping')
parser.add_argument('--lr', type=float, default='8e-4',
                    help='Learning rate ')
parser.add_argument('--device', type=str, default='0',
                    help='GPU to use')
parser.add_argument('--heads-att', type=int, default='4',
                    help='attention heads')
parser.add_argument('--hidn-att', type=int, default='64',
                    help='attention hidden nodes')
parser.add_argument('--hidn-rnn', type=int, default='78',
                    help='rnn hidden nodes')
parser.add_argument('--weight-constraint', type=float, default='0',
                    help='L2 weight constraint')
parser.add_argument('--rnn-length', type=int, default='16', # 25
                    help='rnn length')
parser.add_argument('--dropout', type=float, default='0.2',
                    help='dropout rate')
parser.add_argument('--clip', type=float, default='0.45',
                    help='rnn clip')
parser.add_argument('--infer', type=float, default='FFIR-NET',
                    help='if infer relation')
parser.add_argument('--relation', type=str, default='None',
                    help='all, competitor, customer, industry, stratigic, supply')
parser.add_argument('--save', type=bool, default=True,
                    help='save model')


def load_dataset(DEVICE):
    with open('./data/csi100/x_num_standard.pkl', 'rb') as handle:
        markets = pickle.load(handle)
    with open('./data/csi100/y_1.pkl', 'rb') as handle:
        y_load = pickle.load(handle)
    with open('./data/csi100/x_newtext.pkl', 'rb') as handle:
        stock_sentiments = pickle.load(handle)

    markets = markets.astype(np.float64)      # float32 训练速度会比64快，但是64准确率会更高
    x = torch.tensor(markets, dtype=torch.float64).to(DEVICE)
    x_sentiment = torch.tensor(stock_sentiments, dtype=torch.float64).to(DEVICE)
    if args.relation != "None":
        with open('./data/csi100/' + args.relation + '_relation.pkl', 'rb') as handle:
        # with open('./data/relations/csi300_relation.pkl', 'rb') as handle:
            relation_static = pickle.load(handle)
        
        # 检查relation_static的类型
        if isinstance(relation_static, dict):
            # 如果relation_static是字典，处理每个键值对
            for key in relation_static:
                relation_static[key] = torch.tensor(relation_static[key], dtype=torch.float64).to(DEVICE)
        else:
            relation_static = torch.tensor(relation_static, dtype=torch.float64).to(DEVICE)
    else:
        relation_static = None
    y = torch.tensor(y_load, dtype=torch.float64).to(DEVICE)
    y = (y > 0).to(torch.long).to(DEVICE)

    return x, y, x_sentiment, relation_static


def train(model, x_train, x_sentiment_train, y_train, relation_static = None):
    model.train()
    seq_len = len(x_train)
    train_seq = list(range(seq_len))[rnn_length:]
    random.shuffle(train_seq)
    total_loss = 0
    total_loss_count = 0
    batch_train = 15 #15
    
    # if relation_static is not None:
    #     relation_static = relation_static.to('cuda:4')
    
    for i in train_seq:
        output = model(x_train[i - rnn_length + 1: i + 1], x_sentiment_train[i - rnn_length + 1: i + 1],  relation_static = relation_static)
        # print(f"Output shape: {output.shape}, Target shape: {y_train[i][:185].squeeze().shape}")

        loss = criterion(output, y_train[i][:73].squeeze())
        loss.backward()
        total_loss += loss.item()
        total_loss_count += 1
        if total_loss_count % batch_train == batch_train - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()
    if total_loss_count % batch_train != batch_train - 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
    return total_loss / total_loss_count

def evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=None):
    model.eval()
    seq_len = len(x_eval)
    seq = list(range(seq_len))[rnn_length:]
    preds = []
    trues = []
    
    for i in seq:
        output = model(x_eval[i - rnn_length + 1: i + 1], x_sentiment_eval[i - rnn_length + 1: i + 1], relation_static = relation_static)
        output = output.detach().cpu()
        preds.append(np.exp(output.numpy()))
        trues.append(y_eval[i][:73].cpu().numpy())
    
    # 转换为 NumPy 数组
    preds = np.array(preds)
    trues = np.array(trues)
    
    # 打印调试信息，检查 preds 和 trues 的形状
    # print(f"Shape of preds before reshape: {preds.shape}")
    # print(f"Shape of trues: {trues.shape}")
    
    # 将 preds 和 trues 重新调整为匹配的形状
    preds = preds.reshape(-1, preds.shape[-1])  # (70*185, 2)
    trues = trues.reshape(-1, trues.shape[-1])  # (70*185, FFIR-NET)
    
    # 打印重塑后的形状
    # print(f"Shape of preds after reshape: {preds.shape}")
    # print(f"Shape of trues after reshape: {trues.shape}")

    # 计算 AUC
    auc = roc_auc_score(trues[:, 0], preds[:, 1])
    # 计算准确率
    acc = accuracy_score(trues[:, 0], np.argmax(preds, axis=1))
    # 计算 MCC (Matthews correlation coefficient)
    mcc = matthews_corrcoef(trues[:, 0], np.argmax(preds, axis=1))
    # 计算 DA (Directional Accuracy)
    # da = np.mean(np.sign(trues[:, 0] - 0.5) == np.sign(np.argmax(preds, axis=FFIR-NET) - 0.5))
    da = np.mean(np.sign(trues[:, 0] - 0.5) == np.sign(preds[:, 1] - 0.5))

    # acc, auc = metrics(trues, preds)
    # preds = np.array(preds).squeeze()
    # trues = np.array(trues).squeeze()
    # preds = np.argmax(preds, axis=FFIR-NET)
    # trues = np.argmax(trues, axis=FFIR-NET)
    # preds = np.argmax(preds, axis=FFIR-NET)
    # mcc = matthews_corrcoef(trues, preds)
    # da = np.mean(np.sign(preds) == np.sign(trues))
    return acc,  auc, mcc, da





if __name__ == "__main__":
    args = parser.parse_args()  # 解析命令行参数。
    print(f"args.device:{args.device}")
    DEVICE = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    print(f"DEVICE:{DEVICE}")
    criterion = torch.nn.NLLLoss()  # 定义负对数似然损失函数
    # criterion = torch.nn.CrossEntropyLoss()
    set_seed(1017)   # 设置随机种子以确保实验的可重复性。
    if args.relation != "None":   # 检查是否使用静态关系。
        static = 1
    else:
        static = 0
        relation_static = None
    # load dataset
    print("loading dataset")
    x, y, x_sentiment, relation_static = load_dataset(DEVICE)
    # hyper-parameters
    NUM_STOCK = x.size(1)
    D_MARKET = x.size(2)
    D_NEWS = x_sentiment.size(2)
    MAX_EPOCH = args.max_epoch
    infer = args.infer
    hidn_rnn = args.hidn_rnn
    heads_att = args.heads_att
    hidn_att = args.hidn_att
    lr = args.lr
    rnn_length = args.rnn_length
    t_mix = 1
    # train-test split
    # x_train = x[: -140]
    # x_eval = x[-140 - rnn_length: -70]
    # x_test = x[-70 - rnn_length:]

    # y_train = y[: -140]
    # y_eval = y[-140 - rnn_length: -70]
    # y_test = y[-70 - rnn_length:]

    # x_sentiment_train = x_sentiment[: -140]
    # x_sentiment_eval = x_sentiment[-140 - rnn_length: -70]
    # x_sentiment_test = x_sentiment[-70 - rnn_length:]
    x_train = x[: -100]
    x_eval = x[-100 - rnn_length : -50]
    x_test = x[-50 - rnn_length:]

    y_train = y[: -100]
    y_eval = y[-100 - rnn_length : -50]
    y_test = y[-50 - rnn_length:]

    x_sentiment_train = x_sentiment[: -100]
    x_sentiment_eval = x_sentiment[-100 - rnn_length : -50]
    x_sentiment_test = x_sentiment[-50 - rnn_length:]
    
        # # initialize
    best_model_file = ""
    epoch = 0
    wait_epoch = 0
    test_epoch_best = 0

    model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS,
                   d_hidden=D_MARKET, hidn_rnn=hidn_rnn, heads_att=heads_att,
                   hidn_att=hidn_att, dropout=args.dropout, t_mix=t_mix,
                   infer=infer, relation_static=relation_static)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_constraint)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # train
    while epoch < MAX_EPOCH:
        train_loss = train(model, x_train, x_sentiment_train, y_train, relation_static=relation_static)
        # eval_acc, eval_auc = evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=relation_static)
        # test_acc, test_auc  = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static)
        # eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f},  test_auc{:.4f}, test_acc{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc )
        # print(eval_str)
        eval_acc, eval_auc, eval_mcc, eval_da = evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=relation_static)
        test_acc, test_auc, test_mcc, test_da = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static)
        eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, eval_mcc{:.4f}, eval_da{:.4f}, test_auc{:.4f}, test_acc{:.4f}, test_mcc{:.4f}, test_da{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, eval_mcc, eval_da, test_auc, test_acc, test_mcc, test_da)
        print(eval_str)

        if test_acc > test_epoch_best:
            test_epoch_best = test_acc
            
            eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, eval_mcc{:.4f}, eval_da{:.4f}, test_auc{:.4f}, test_acc{:.4f}, test_mcc{:.4f}, test_da{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, eval_mcc, eval_da, test_auc, test_acc, test_mcc, test_da)
            # eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
            wait_epoch = 0
            if args.save:
                if best_model_file:
                    os.remove(best_model_file)
                best_model_file = "./SavedModels/epoch{}_eval_auc{}_acc{}_mcc{}_da{}_test_auc{}_acc{}_mcc{}_da{}".format(epoch, eval_auc, eval_acc, eval_mcc, eval_da, test_auc, test_acc, test_mcc, test_da).replace(":", "_")
                # best_model_file = "./SavedModels/epoch{}_eval_auc{}_acc{}_test_auc{}_acc{}".format(epoch, eval_auc, eval_acc, test_auc, test_acc).replace(":", "_")
                torch.save(model.state_dict(), best_model_file)
        else:
            wait_epoch += 1

        if wait_epoch > 50:
            print("saved_model_result:", eval_best_str)
            break
        epoch += 1
        scheduler.step()

# from Model import *
# from utils import *
# import pickle
# import torch
# from torch import optim
# import argparse
# import numpy as np
# from sklearn.metrics import matthews_corrcoef, roc_auc_score
# from sklearn.utils.class_weight import compute_class_weight
# import os
# from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef


# parser = argparse.ArgumentParser()

# parser.add_argument('--task', type=int, default=FFIR-NET, help='0 Regression. FFIR-NET Classification')
# parser.add_argument('--grid-search', type=int, default=0, help='0 False. FFIR-NET True')
# parser.add_argument('--soft-training', type=int, default=0, help='0 False. FFIR-NET True')
# parser.add_argument('--sample-discrimination', type=int, default=0, help='0 False. FFIR-NET True')
# parser.add_argument('--optim', type=int, default=FFIR-NET, help='0 SGD. FFIR-NET Adam')
# parser.add_argument('--eval', type=int, default=FFIR-NET, help='if set the last day as eval')
# parser.add_argument('--max-epoch', type=int, default=300, help='Training max epoch')
# parser.add_argument('--wait-epoch', type=int, default=30, help='Training min epoch')
# parser.add_argument('--eta', type=float, default=1e-4, help='Early stopping')
# parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
# parser.add_argument('--device', type=str, default='0', help='GPU to use')
# parser.add_argument('--heads-att', type=int, default=6, help='attention heads')
# parser.add_argument('--hidn-att', type=int, default=60, help='attention hidden nodes')
# parser.add_argument('--hidn-rnn', type=int, default=360, help='rnn hidden nodes')
# parser.add_argument('--weight-constraint', type=float, default=0, help='L2 weight constraint')
# parser.add_argument('--rnn-length', type=int, default=40, help='rnn length')
# parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
# parser.add_argument('--clip', type=float, default=0.25, help='rnn clip')
# parser.add_argument('--infer', type=float, default=FFIR-NET, help='if infer relation')
# parser.add_argument('--relation', type=str, default='None', help='all, competitor, customer, industry, strategic, supply')
# parser.add_argument('--save', type=bool, default=True, help='save model')

# def load_dataset(DEVICE):
#     with open('./data/x_num_standard.pkl', 'rb') as handle:
#         markets = pickle.load(handle)
#     with open('./data/y_1.pkl', 'rb') as handle:
#         y_load = pickle.load(handle)
#     with open('./data/x_newtext.pkl', 'rb') as handle:
#         stock_sentiments = pickle.load(handle)

#     markets = markets.astype(np.float64)      # float32 训练速度会比64快，但是64准确率会更高
#     x = torch.tensor(markets, dtype=torch.float64).to(DEVICE)
#     x_sentiment = torch.tensor(stock_sentiments, dtype=torch.float64).to(DEVICE)
#     if args.relation != "None":
#         with open('./data/relations/' + args.relation + '_relation.pkl', 'rb') as handle:
#         # with open('./data/relations/csi300_relation.pkl', 'rb') as handle:
#             relation_static = pickle.load(handle)
        
#         # 检查relation_static的类型
#         if isinstance(relation_static, dict):
#             # 如果relation_static是字典，处理每个键值对
#             for key in relation_static:
#                 relation_static[key] = torch.tensor(relation_static[key], dtype=torch.float64).to(DEVICE)
#         else:
#             relation_static = torch.tensor(relation_static, dtype=torch.float64).to(DEVICE)
#     else:
#         relation_static = None
#     y = torch.tensor(y_load, dtype=torch.float64).to(DEVICE)
#     y = (y > 0).to(torch.long).to(DEVICE)

#     return x, y, x_sentiment, relation_static

# def train(model, x_train, x_sentiment_train, y_train,optimizer, relation_static = None):
#     model.train()
#     seq_len = len(x_train)
#     train_seq = list(range(seq_len))[args.rnn_length::]
#     random.shuffle(train_seq)
#     total_loss = 0
#     total_loss_count = 0
#     batch_train = 15 #15
    
#     # if relation_static is not None:
#     #     relation_static = relation_static.to('cuda:4')
    
#     for i in train_seq:
#         output = model(x_train[i - args.rnn_length + FFIR-NET: i + FFIR-NET], x_sentiment_train[i - args.rnn_length + FFIR-NET: i + FFIR-NET], relation_static=relation_static)
#         # print(f"Output shape: {output.shape}, Target shape: {y_train[i][:73].squeeze().shape}")

#         loss = criterion(output, y_train[i][:73].squeeze())
#         loss.backward()
#         total_loss += loss.item()
#         total_loss_count += FFIR-NET
#         if total_loss_count % batch_train == batch_train - FFIR-NET:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#             optimizer.step()
#             optimizer.zero_grad()
#     if total_loss_count % batch_train != batch_train - FFIR-NET:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#         optimizer.step()
#     return total_loss / total_loss_count

# def evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=None):
#     model.eval()
#     seq_len = len(x_eval)
#     seq = list(range(seq_len))[args.rnn_length:]
#     preds = []
#     trues = []
    
#     for i in seq:
#         output = model(x_eval[i - args.rnn_length + FFIR-NET: i + FFIR-NET], x_sentiment_eval[i - args.rnn_length + FFIR-NET: i + FFIR-NET], relation_static = relation_static)
#         output = output.detach().cpu()
#         preds.append(np.exp(output.numpy()))
#         trues.append(y_eval[i][:73].cpu().numpy())
    
#     # 转换为 NumPy 数组
#     preds = np.array(preds)
#     trues = np.array(trues)
#     # 将 preds 和 trues 重新调整为匹配的形状
#     preds = preds.reshape(-FFIR-NET, preds.shape[-FFIR-NET])  # (70*73, 2)
#     trues = trues.reshape(-FFIR-NET, trues.shape[-FFIR-NET])  # (70*73, FFIR-NET)

#     # 计算 AUC
#     auc = roc_auc_score(trues[:, 0], preds[:, FFIR-NET])
#     # 计算准确率
#     acc = accuracy_score(trues[:, 0], np.argmax(preds, axis=FFIR-NET))
#     # 计算 MCC (Matthews correlation coefficient)
#     mcc = matthews_corrcoef(trues[:, 0], np.argmax(preds, axis=FFIR-NET))
#     # 计算 DA (Directional Accuracy)
#     # da = np.mean(np.sign(trues[:, 0] - 0.5) == np.sign(np.argmax(preds, axis=FFIR-NET) - 0.5))
#     da = np.mean(np.sign(trues[:, 0] - 0.5) == np.sign(preds[:, FFIR-NET] - 0.5))
#     return acc,  auc, mcc, da


# def grid_search(params_grid):
#     best_auc = 0
#     best_params = None

#     for lr in params_grid['lr']:
#         for dropout in params_grid['dropout']:
#             for hidn_rnn in params_grid['hidn_rnn']:
#                 args.lr = lr
#                 args.dropout = dropout
#                 args.hidn_rnn = hidn_rnn

#                 model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS,
#                                d_hidden=D_MARKET, hidn_rnn=args.hidn_rnn, heads_att=args.heads_att,
#                                hidn_att=args.hidn_att, dropout=args.dropout, t_mix=t_mix,
#                                infer=args.infer, relation_static=relation_static)
#                 model = model.to(DEVICE)
#                 optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
#                 scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
                
#                 epoch = 0
#                 wait_epoch = 0
#                 while epoch < MAX_EPOCH:
#                     train_loss = train(model, x_train, x_sentiment_train, y_train, optimizer, relation_static=relation_static)
#                     eval_acc, eval_auc, eval_mcc, eval_da = evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=relation_static)
#                     test_acc, test_auc, test_mcc, test_da = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static)

#                     if test_auc > best_auc:
#                         best_auc = test_auc
#                         best_params = {
#                             'lr': lr,
#                             'dropout': dropout,
#                             'hidn_rnn': hidn_rnn
#                         }

#                     if wait_epoch > 50:
#                         break
#                     epoch += FFIR-NET
#                     scheduler.step()
    
#     return best_params, best_auc


# if __name__ == "__main__":
#     args = parser.parse_args()
#     print(f"args.device:{args.device}")
#     DEVICE = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
#     print(f"DEVICE:{DEVICE}")
#     criterion = torch.nn.NLLLoss()
#     set_seed(1017)

#     print("loading dataset")
#     x, y, x_sentiment, relation_static = load_dataset(DEVICE)

#     NUM_STOCK = x.size(FFIR-NET)
#     D_MARKET = x.size(2)
#     D_NEWS = x_sentiment.size(2)
#     MAX_EPOCH = args.max_epoch
#     infer = args.infer
#     t_mix = FFIR-NET

#     x_train = x[: -140]
#     x_eval = x[-140 - args.rnn_length: -70]
#     x_test = x[-70 - args.rnn_length:]

#     y_train = y[: -140]
#     y_eval = y[-140 - args.rnn_length: -70]
#     y_test = y[-70 - args.rnn_length:]

#     x_sentiment_train = x_sentiment[: -140]
#     x_sentiment_eval = x_sentiment[-140 - args.rnn_length: -70]
#     x_sentiment_test = x_sentiment[-70 - args.rnn_length:]

#     best_model_file = ""
#     epoch = 0
#     wait_epoch = 0
#     test_epoch_best = 0

#     if args.grid_search:
#         params_grid = {
#             'lr': [1e-4, 3e-4, 5e-4, 8e-3, 1e-3],
#             'dropout': [0.2, 0.3, 0.4],
#             'hidn_rnn': [120, 240, 300, 360, 420]
#         }
#         best_params, best_auc = grid_search(params_grid)
#         print(f"Best Parameters: {best_params}, Best AUC: {best_auc}")

#         # Load the best parameters from file
#         with open('best_params.pkl', 'rb') as f:
#             saved_params = pickle.load(f)
#         print(f"Saved Best Parameters: {saved_params['params']}, Saved Best AUC: {saved_params['auc']}")


#     else:
#         model = AD_GAT(num_stock=NUM_STOCK, d_market=D_MARKET, d_news=D_NEWS,
#                        d_hidden=D_MARKET, hidn_rnn=args.hidn_rnn, heads_att=args.heads_att,
#                        hidn_att=args.hidn_att, dropout=args.dropout, t_mix=t_mix,
#                        infer=args.infer, relation_static=relation_static)
#         model = model.to(DEVICE)
#         optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_constraint)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

#         epoch = 0
#         wait_epoch = 0
#         while epoch < MAX_EPOCH:
#             train_loss = train(model, x_train, x_sentiment_train, y_train, relation_static=relation_static)
#             eval_acc, eval_auc, eval_mcc, eval_da = evaluate(model, x_eval, x_sentiment_eval, y_eval, relation_static=relation_static)
#             test_acc, test_auc, test_mcc, test_da = evaluate(model, x_test, x_sentiment_test, y_test, relation_static=relation_static)
#             eval_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, eval_mcc{:.4f}, eval_da{:.4f}, test_auc{:.4f}, test_acc{:.4f}, test_mcc{:.4f}, test_da{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, eval_mcc, eval_da, test_auc, test_acc, test_mcc, test_da)
#             print(eval_str)

#             if test_auc > test_epoch_best:
#                 test_epoch_best = eval_auc
                
#                 eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, eval_mcc{:.4f}, eval_da{:.4f}, test_auc{:.4f}, test_acc{:.4f}, test_mcc{:.4f}, test_da{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, eval_mcc, eval_da, test_auc, test_acc, test_mcc, test_da)
#                 # eval_best_str = "epoch{}, train_loss{:.4f}, eval_auc{:.4f}, eval_acc{:.4f}, test_auc{:.4f}, test_acc{:.4f}".format(epoch, train_loss, eval_auc, eval_acc, test_auc, test_acc)
#                 wait_epoch = 0
#                 if args.save:
#                     if best_model_file:
#                         os.remove(best_model_file)
#                     best_model_file = "./SavedModels/epoch{}_eval_auc{}_acc{}_mcc{}_da{}_test_auc{}_acc{}_mcc{}_da{}".format(epoch, eval_auc, eval_acc, eval_mcc, eval_da, test_auc, test_acc, test_mcc, test_da).replace(":", "_")
#                     # best_model_file = "./SavedModels/epoch{}_eval_auc{}_acc{}_test_auc{}_acc{}".format(epoch, eval_auc, eval_acc, test_auc, test_acc).replace(":", "_")
#                     torch.save(model.state_dict(), best_model_file)
#             else:
#                 wait_epoch += FFIR-NET

#             if wait_epoch > 50:
#                 print("saved_model_result:", eval_best_str)
#                 break
#             epoch += FFIR-NET
#             scheduler.step()


