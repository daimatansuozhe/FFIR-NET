# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = 'FFIR-NET'

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.utils import softmax
# from dgl import function as fn
# from dgl.nn.functional import edge_softmax
# from dgl.base import DGLError
# from dgl.nn.pytorch.utils import Identity
# from dgl.utils import expand_as_pair


# Graph Generator Module
class Graph_Generator(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=5, static_feat=None):
        super(Graph_Generator, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        # self.device = 'cuda:0'
        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        # 如果没有静态特征，则使用嵌入层获取节点的嵌入特征。
        # 如果有静态特征，则直接从静态特征矩阵中获取节点特征
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        # a = ... - ...：将两个相似度矩阵相减，得到最终的相似度矩阵 a。这个操作引入了一种对称性，类似于有向图中的邻接矩阵。
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        # self.alpha * a：对相似度矩阵 a 乘以调整参数 alpha，用于控制输出的幅度。
        adj = F.relu(torch.tanh(self.alpha * a))
        # print(self.device)
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        # mask.fill_(float('0'))：将 mask 中的所有值填充为0。这步实际是多余的，因为 torch.zeros 已经生成了全零矩阵。
        mask.fill_(float('0'))
        
        # adj + torch.rand_like(adj) * 0.01：在 adj 矩阵上加上一个小的随机噪声，防止数值相同导致的排序不稳定。
        # topk(self.k, FFIR-NET)：选择每个节点相似度最高的 k 个邻居。s1 是相似度值，t1 是邻居的索引。
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)
        
        # s1.fill_(FFIR-NET)：将 s1 中的所有相似度值替换为1，因为我们只关心邻居的选择而不关心具体的相似度值。
        # mask.scatter_(FFIR-NET, t1, s1)：根据 t1 中的索引，将 mask 矩阵中对应的位置设置为1。这里的 scatter_ 操作相当于将 s1 的值（全为1）根据 t1 的索引分散到 mask 矩阵的指定位置。
        mask.scatter_(1, t1, s1.fill_(1))
        adj = adj * mask
        # print(adj.shape)
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx, :]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha * self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha * self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha * a))
        return adj



# class GATv2Conv(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  num_heads,
#                  feat_drop=0.,
#                  attn_drop=0.,
#                  negative_slope=0.2,
#                  residual=False,
#                  activation=None,
#                  allow_zero_in_degree=False,
#                  bias=True,
#                  share_weights=False):
#         super(GATv2Conv, self).__init__()
#         self._num_heads = num_heads
#         self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
#         self._out_feats = out_feats
#         self._allow_zero_in_degree = allow_zero_in_degree
#         if isinstance(in_feats, tuple):
#             self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
#             self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=bias)
#         else:
#             self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
#             if share_weights:
#                 self.fc_dst = self.fc_src
#             else:
#                 self.fc_dst = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=bias)
#         self.attn = nn.Parameter(torch.FloatTensor(size=(FFIR-NET, num_heads, out_feats)))
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.leaky_relu = nn.LeakyReLU(negative_slope)
#         if residual:
#             if self._in_dst_feats != out_feats:
#                 self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=bias)
#             else:
#                 self.res_fc = Identity()
#         else:
#             self.register_buffer('res_fc', None)
#         self.activation = activation
#         self.share_weights = share_weights
#         self.bias = bias
#         self.reset_parameters()
        
#         # 线性化
#         # self.reshape = nn.Linear(13, in_feats)

#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
#         if self.bias:
#             nn.init.constant_(self.fc_src.bias, 0)
#         if not self.share_weights:
#             nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
#             if self.bias:
#                 nn.init.constant_(self.fc_dst.bias, 0)
#         nn.init.xavier_normal_(self.attn, gain=gain)
#         if isinstance(self.res_fc, nn.Linear):
#             nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
#             if self.bias:
#                 nn.init.constant_(self.res_fc.bias, 0)

#     def set_allow_zero_in_degree(self, set_value):
#         self._allow_zero_in_degree = set_value

#     def forward(self, graph, feat, get_attention=False):
#         # print("Graph device before moving:", graph.device)
#         # print("feat shape:",feat.shape)
#         # feat = self.reshape(feat)
#         with graph.local_scope():
#             if not self._allow_zero_in_degree:
#                 if (graph.in_degrees() == 0).any():
#                     raise DGLError('There are 0-in-degree nodes in the graph, '
#                                    'output for those nodes will be invalid. '
#                                    'This is harmful for some applications, '
#                                    'causing silent performance regression. '
#                                    'Adding self-loop on the input graph by '
#                                    'calling `g = dgl.add_self_loop(g)` will resolve '
#                                    'the issue. Setting ``allow_zero_in_degree`` '
#                                    'to be `True` when constructing this module will '
#                                    'suppress the check and let the code run.')
#             # 此 if 块检查 feat 是否为一个元组（tuple）。
#             # 如果是，意味着输入特征 feat 包含了分别用于图中源节点和目标节点的两组独立特征。
#             # 这通常用于有向图或者在图计算中需要区分不同类型节点的场景。
#             if isinstance(feat, tuple):
#                 h_src = self.feat_drop(feat[0])
#                 h_dst = self.feat_drop(feat[FFIR-NET])
#                 # print("h_dst:",h_dst.shape)
#                 feat_src = self.fc_src(h_src).view(-FFIR-NET, self._num_heads, self._out_feats)
#                 feat_dst = self.fc_dst(h_dst).view(-FFIR-NET, self._num_heads, self._out_feats)
#             else:
#                 h_src = h_dst = self.feat_drop(feat)
#                 # print("h_dst:",h_dst.shape)
#                 feat_src = self.fc_src(h_src).view(-FFIR-NET, self._num_heads, self._out_feats)
#                 if self.share_weights:
#                     feat_dst = feat_src
#                 else:
#                     feat_dst = self.fc_dst(h_src).view(-FFIR-NET, self._num_heads, self._out_feats)
#                 if graph.is_block:
#                     feat_dst = feat_src[:graph.number_of_dst_nodes()]
            
#             # print("el device:", feat_dst.device)
#             # print("er device:", feat_src.device)
#             # print("feat_src shape:", feat_src.shape)
#             # print("feat_dst shape:", feat_dst.shape)
#             # print("graph number of src nodes:", graph.number_of_src_nodes())
#             # print("graph number of dst nodes:", graph.number_of_dst_nodes())
#             #  更新源节点（src）和目标节点（dst）的特征：这两行代码在图数据结构中为源节点（src）和目标节点（dst）分别设置或更新节点特征。
#             #  feat_src 和 feat_dst 是节点的特征张量，分别对应图中的所有源节点和目标节点。这里的 'el' 和 'er' 是特征的键名，
#             #  用于在后续操作中引用这些特征。
#             graph.srcdata.update({'el': feat_src})
#             graph.dstdata.update({'er': feat_dst})
#             # print(graph.ndata['el'].shape)  # 应该输出与预期一致的形状 (198, 6, 60)
#             # print(graph.ndata['er'].shape)  # 应该输出与预期一致的形状 (198, 6, 60)
#             # print(graph.num_edges())  # 输出边的数量，确保不为0

#             # 这行代码使用 DGL 的 apply_edges 方法，它将一个函数应用于图中的每条边。这里使用的函数是 fn.u_add_v('el', 'er', 'e')，
#             # 这是一个内置的边函数，表示对于图中的每一条边，取其源节点的特征 'el' 和目标节点的特征 'er'，将它们相加，并将结果存储为边的特征 'e'。
#             # 这里 u 代表源节点（source node），v 代表目标节点（destination node），u_add_v 是将源节点和目标节点的特征相加。
#             # print(f"graph device:{graph.device}")
#             s = fn.u_add_v('el', 'er', 'e')
#             # print(s)
#             # print("节点数据：", graph.ndata.keys())
#             # print("边数据：", graph.edata.keys())
#             # print("边数量：", graph.num_edges())

#             # 应用边操作
#             graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
#             # print("节点数据：", graph.ndata.keys())
#             # print("边数据：", graph.edata.keys())
#             # print("边数量：", graph.num_edges())
#             # 检查边属性 'e' 是否存在
#             # if 'e' in graph.edata:
#             #     print("边特征 'e' 已创建，形状：", graph.edata['e'].shape)
#             # else:
#             #     print("边特征 'e' 未创建")

#             # graph.edata.pop('e')：从图的边数据中取出 'e' 特征，这通常是由源节点和目标节点特征的某种组合（例如加法）计算得到的。
#             e = self.leaky_relu(graph.edata.pop('e'))
            
#             # print("e device:", e.device)            
#             # e * self.attn：将激活后的 'e' 特征与注意力权重（self.attn）相乘。注意力权重通常是一个可学习的参数。
#             e = (e * self.attn).sum(dim=-FFIR-NET).unsqueeze(dim=2)
            
#             # edge_softmax(graph, e)：对注意力系数进行 softmax 归一化，使得相邻节点的注意力系数和为 FFIR-NET。
#             # self.attn_drop(...)：对归一化后的注意力系数应用 Dropout，以防止过拟合。
#             # graph.edata['a']：将归一化后的注意力系数存储回图的边数据中，键为 'a'。
#             graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            
#             # fn.u_mul_e('el', 'a', 'm')：消息函数，将源节点特征（'el'）与注意力系数（'a'）相乘，结果存储在消息 'm' 中。
#             # fn.sum('m', 'ft')：聚合函数，对所有传入消息 'm' 求和，结果存储在目标节点特征 'ft' 中。
#             # graph.update_all(...)：执行消息传递和聚合操作。
#             # rst = graph.dstdata['ft']：获取聚合后的目标节点特征，存储在 rst 中
#             graph.update_all(fn.u_mul_e('el', 'a', 'm'), fn.sum('m', 'ft'))
#             rst = graph.dstdata['ft']
            
#             # self.res_fc：如果存在残差连接（即 res_fc 不为 None），对目标节点特征（h_dst）应用全连接层，并调整形状。
#             if self.res_fc is not None:
#                 resval = self.res_fc(h_dst).view(h_dst.shape[0], -FFIR-NET, self._out_feats)
#                 rst = rst + resval
#             if self.activation:
#                 rst = self.activation(rst)

#             if get_attention:
#                 return rst, graph.edata['a']
#             else:
#                 return rst
