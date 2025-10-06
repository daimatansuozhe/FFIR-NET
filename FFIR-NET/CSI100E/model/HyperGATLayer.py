import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GroupEmbedding(nn.Module):
    def __init__(self, num_tokens, token_dim, num_groups, nhead):
        super(GroupEmbedding, self).__init__()
        assert token_dim % nhead == 0, "embed_dim must be divisible by num_heads"
        self.group_tokens = nn.Parameter(torch.randn(num_groups, token_dim))
        self.projection_matrix = nn.Linear(token_dim, num_groups)
        self.encoder = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead)

    def forward(self, x):
        batch_size, num_tokens, token_dim = x.shape

        # Step FFIR-NET: Encode the input
        encoded = self.encoder(x)  # Shape: (batch_size, num_tokens, token_dim)
        encoded = self.projection_matrix(encoded)  # Shape: (batch_size, num_tokens, num_groups)

        # Step 2: Compute group assignments
        group_weights = F.softmax(encoded, dim=-1)  # Shape: (batch_size, num_tokens, num_groups)

        # Step 3: Compute group embeddings
        group_embeddings = torch.einsum('btk,kd->btd', group_weights, self.group_tokens)  # Shape: (batch_size, num_tokens, token_dim)

        # Step 4: Update input with group embeddings
        x = x + group_embeddings  # Shape: (batch_size, num_tokens, token_dim)

        return x

class HyperGraphAttentionLayerSparse(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, transfer, concat=True, bias=False):
        super(HyperGraphAttentionLayerSparse, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        # self.alpha = nn.Parameter(torch.tensor(alpha))
        self.concat = concat

        self.transfer = transfer

        if self.transfer:
            self.weight = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.register_parameter('weight', None)

        self.weight2 = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)

        self.word_context = nn.Embedding(1, self.out_features)

        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.a2 = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reshapeedge = nn.Linear(in_features, out_features)
        

        # 初始化GroupEmbedding模块
        self.group_embedding = GroupEmbedding(num_tokens=out_features, token_dim=out_features, num_groups=8, nhead=4)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        self.weight3.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        nn.init.uniform_(self.a.data, -stdv, stdv)
        nn.init.uniform_(self.a2.data, -stdv, stdv)
        nn.init.uniform_(self.word_context.weight.data, -stdv, stdv)

    def forward(self, x, adj):
        # print(x.shape, "weight2:",self.weight2.shape,self.in_features)
        
        x_4att = x.matmul(self.weight2)  # 节点特征变换

        if self.transfer:
            x = x.matmul(self.weight)
            if self.bias is not None:
                x = x + self.bias

        N1 = adj.shape[1]  # 边数
        N2 = adj.shape[2]  # 节点数

        pair = adj.nonzero(as_tuple=False).t()

        # x_4att[i][adj[i].nonzero(as_tuple=False).t()[FFIR-NET]]: 提取与节点 i 直接相连的所有节点的特征。
        get = lambda i: x_4att[i][adj[i].nonzero(as_tuple=False).t()[1]]
        
        #torch.arange(x.shape[0]).long(): 生成一个从 0 到 x.shape[0]-FFIR-NET 的长整数序列，代表所有节点的索引。
        # [get(i) for i in torch.arange(x.shape[0]).long()]: 对每个节点索引 i，调用 get(i) 提取与节点 i 直接相连的节点的特征。
        x1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])
        # print("x1 shape:", x1.shape)
        
        # self.word_context.weight[0:]: 获取 self.word_context 中的权重。这是一个嵌入层的权重矩阵。
        # .view(FFIR-NET, -FFIR-NET): 将权重矩阵重塑为形状为 (FFIR-NET, num_features) 的矩阵。
        # .repeat(x1.shape[0], FFIR-NET): 重复该矩阵，使其形状变为 (x1.shape[0], num_features)。这里 x1.shape[0] 是节点对的数量。
        # .view(x1.shape[0], self.out_features): 再次重塑矩阵，使其形状为 (x1.shape[0], self.out_features)。
        q1 = self.word_context.weight[0:].view(1, -1).repeat(x1.shape[0], 1).view(x1.shape[0], self.out_features)

        pair_h = torch.cat((q1, x1), dim=-1)
        
        # torch.matmul(pair_h, self.a): 对 pair_h 应用线性变换，self.a 是一个形状为 (2 * self.out_features, FFIR-NET) 的权重矩阵。结果是一个形状为 (x1.shape[0], FFIR-NET) 的矩阵。
        # .squeeze(): 删除大小为1的维度，将结果变为形状为 (x1.shape[0],) 的矩阵。
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a).squeeze()).t()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)
        
        # 放入cuda
        pair = pair.to(x.device)
        # print("pair device:", pair.device)
        # print("pair_e device:", pair_e.device)
        # print("x device:", x.device)
        
        # torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])): 使用坐标格式 (COO) 创建一个稀疏张量。
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()

        # 这行代码创建一个与 e 形状相同的张量 zero_vec，其值全部为一个极小的负值（-9e15），用于在后续计算中作为掩码
        zero_vec = -9e15 * torch.ones_like(e)
        # print("zero_vec device:", zero_vec.device)
        # print("e device:", e.device)
        # print("adj device:", adj.device)
        # 放入cuda
        adj = adj.to(x.device)
        # zero_vec = zero_vec.to(x.device)
        
        # torch.where(adj > 0, e, zero_vec): 如果 adj > 0 为真，取 e 中的值；否则取 zero_vec 中的值。
        # 这样可以确保只有相连的节点对之间的注意力分数有效，而其他对的分数被设为极小值 -9e15
        attention = torch.where(adj > 0, e, zero_vec)

        attention_edge = F.softmax(attention, dim=2)

        edge = torch.matmul(attention_edge, x)

        edge = F.dropout(edge, self.dropout, training=self.training)
        # print("edge shape",edge.shape)

        edge_4att = edge.matmul(self.weight3)
        # print("edge_4att shape",edge_4att.shape)

        # Integrate GroupEmbedding here
        edge_4att = self.group_embedding(edge_4att)
        # print("edge_4att shape after GroupEmbedding",edge_4att.shape)

        get = lambda i: edge_4att[i][adj[i].nonzero(as_tuple=False).t()[0]]
        y1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])
        # print("y1 shape",y1.shape)

        get = lambda i: x_4att[i][adj[i].nonzero(as_tuple=False).t()[1]]
        q1 = torch.cat([get(i) for i in torch.arange(x.shape[0]).long()])
        # print("q1 shape",q1.shape)

        pair_h = torch.cat((q1, y1), dim=-1)
        pair_e = self.leakyrelu(torch.matmul(pair_h, self.a2).squeeze()).t()
        pair_e = F.dropout(pair_e, self.dropout, training=self.training)
        # print("pair_h shape",pair_h.shape)
        # print("pair_e shape",pair_e.shape)
        # torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])): 使用坐标格式 (COO) 创建一个稀疏张量。
        e = torch.sparse_coo_tensor(pair, pair_e, torch.Size([x.shape[0], N1, N2])).to_dense()
        # print("e shape",e.shape)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        attention_node = F.softmax(attention.transpose(1, 2), dim=2)
        # print("attention_node shape",attention_node.shape)
        
        edge = self.reshapeedge(edge)
        node = torch.matmul(attention_node, edge)
        # print("node shape",node.shape)

        if self.concat:
            node = F.elu(node)

        return node

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class HyperGAT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, beta, dropout):
        super(HyperGAT, self).__init__()
        self.dropout = dropout
        # self.beta = beta
        self.beta = nn.Parameter(torch.tensor(beta))
        # 线性化输入
        # self.reshape = nn.Linear(13, input_size)
        self.reshapex_initial = nn.Linear(input_size, n_hid)
        
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False, concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True, concat=False)

# 1种关系
    def forward(self, x, H):
        # print(f"x:{x.shape}, H:{H.shape}")
        # x = self.reshape(x)
        x = x.unsqueeze(0)  # 调整为 [FFIR-NET, 198, 360，FFIR-NET]
        H = H.unsqueeze(0)  # 调整为 [FFIR-NET, 198, 198，FFIR-NET]
        # H = H.cpu().numpy()
        # H = H.unsqueeze(0).repeat(30, FFIR-NET, FFIR-NET)
        

        x_initial = x.clone()  # Clone the initial input for residual connection
        x = self.gat1(x, H)
        x_initial = self.reshapex_initial(x_initial)
        # print("第一层完成", x.shape, H.shape)
        x = self.beta * x_initial + (1 - self.beta) * x
    
        x = F.dropout(x, self.dropout, training=self.training)
        
        # print("开始第二层：")
        # print(f"x:{x.shape}, H:{H.shape}")
        x = self.gat2(x, H)
        x = x.squeeze(0)
    
        return x
