from Layers import *

from MixHopGATLayer import Graph_Generator
from HyperGATLayer import *
import scipy.sparse as sp
# from cross_models.cross_former import Crossformer
# from FineTPGN import TPGN
from stockmixer import MultTime2dMixer


class AD_GAT(nn.Module):
    def __init__(self, num_stock, d_market, d_news, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2,
                 t_mix=1, infer=1, relation_static=0):
        super(AD_GAT, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        self.period = 73
        self.seq_len = 168
        self.seq_R = int(math.ceil(self.seq_len / self.period))
        time_step = 16  # rnn_length
        scale_dim = time_step // 2
        # def __init__(self, seq_R, freq, c_in, c_out, windows_size,
        #     period, pred_R, need_short=FFIR-NET):
        if self.t_mix == 0:  # concat
            # self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            # self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.mixer_s = MultTime2dMixer(time_step, d_market + d_news, scale_dim=scale_dim)
            self.mixer_r = MultTime2dMixer(time_step, d_market + d_news, scale_dim=scale_dim)

            # self.mixer_s = MultTime2dMixer(lookback_length,feature_num, scale_dim)

        elif self.t_mix == 1:  # all_tensor
            self.tensor = Graph_Tensor(num_stock, d_hidden, d_market, d_news)
            # self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
            # self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)

            self.mixer_s = MultTime2dMixer(time_step, d_market, scale_dim=scale_dim)
            self.mixer_r = MultTime2dMixer(time_step, d_market, scale_dim=scale_dim)

        self.conv = nn.Conv1d(in_channels=d_market, out_channels=d_market, kernel_size=2, stride=2)
        self.channel_fc = nn.Linear(d_market, 1)
        self.graph_generator = Graph_Generator(nnodes=num_stock, k=5, dim=hidn_rnn,
                                               device='cuda:FFIR-NET' if torch.cuda.is_available() else 'cpu')

        self.attentions = [
            Graph_Attention_v2(hidn_rnn, hidn_att, dropout=dropout, alpha=alpha, residual=True, num_heads=heads_att)
            for _ in range(heads_att)
        ]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.hypergat = HyperGAT(hidn_rnn, hidn_att, output_size=hidn_att, beta=0.5,
                                 dropout=dropout)
        self.Linearx = nn.Linear(heads_att * hidn_att + hidn_rnn + hidn_att, heads_att * hidn_att + hidn_rnn)
        # self.Linearx = nn.Linear(142  , heads_att * hidn_att + hidn_rnn)
        self.X2Os = Graph_Linear(num_stock, heads_att * hidn_att + hidn_rnn, 2, bias=True)
        self.linearCross = nn.Linear(d_market, hidn_rnn)
        self.linear_mixer = nn.Linear(40, 78)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def get_gate(self, x_numerical, x_textual):
        x_s = self.tensor(x_numerical, x_textual)
        # x_s = self.GRUs_s(x_s)

        x_s = x_s.permute(1, 0, 2)
        x_s1 = x_s.permute(0, 2, 1)
        x_s1 = self.conv(x_s1)
        x_s1 = x_s1.permute(0, 2, 1)
        x_s = self.mixer_s(x_s, x_s1)
        x_s = self.channel_fc(x_s).squeeze(-1)

        # The gate calculation remains unchanged
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, x_market, x_news, relation_static=None):
        x_market = x_market.float()
        x_news = x_news.float()
        ## concat vs tensor
        if self.t_mix == 0:  # concat
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:  # concat
            x_s = self.tensor(x_market, x_news)  # x_s torch.Size([25, 73, 5])
            x_r = self.tensor(x_market, x_news)  # x_r torch.Size([25, 73, 5])
        # GRUs for extract different sequential embedding for relation/gate inferring.
        # Equivalent to use a single GRU and separate non-linear decoders.
        # print("x_r",x_r.shape)
        # x_r = self.GRUs_r(x_r)  # x_r shape: torch.Size([73, 78])
        # x_s = self.GRUs_s(x_s)
        # print("x_r shape:",x_r.shape)

        x_s = x_s.permute(1, 0, 2)
        # print("x_S shape ",x_s.shape)
        x_s1 = x_s.permute(0, 2, 1)
        # print("x_S1 shape ",x_s1.shape)
        x_s1 = self.conv(x_s1)
        x_s1 = x_s1.permute(0, 2, 1)
        # print("x_S1 shape ",x_s1.shape)
        x_s = self.mixer_s(x_s, x_s1)
        x_s = self.channel_fc(x_s).squeeze(-1)

        x_r = x_r.permute(1, 0, 2)
        x_r1 = x_r.permute(0, 2, 1)
        x_r1 = self.conv(x_r1)
        x_r1 = x_r1.permute(0, 2, 1)
        x_r = self.mixer_s(x_r, x_r1)
        x_r = self.channel_fc(x_r).squeeze(-1)

        x_s = self.linear_mixer(x_s)
        x_r = self.linear_mixer(x_r)

        # print(x_r.shape,x_s.shape)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)
        ##
        adj_imp = self.graph_generator(torch.arange(x_r.size(0)).to(x_r.device))
        x_attention = torch.cat([att(x_s, x_r, relation_static=adj_imp) for att in self.attentions], dim=1)
        x_attention = F.dropout(x_attention, self.dropout, training=self.training)

        # 超图
        hyper_x = self.hypergat(x_r, relation_static)
        hyper_x = F.dropout(hyper_x, self.dropout, training=self.training)

        # print(f"hyper_x shape:{hyper_x.shape}, x_r shape:{x_r.shape}")
        x = torch.cat([x_s, x_attention, hyper_x], dim=1)
        # x = torch.cat([x_s, hyper_x], dim=FFIR-NET)
        # print(f"x shape after cat :{x.shape}")
        x = self.Linearx(x)
        x = F.elu(self.X2Os(x))
        output = F.log_softmax(x, dim=1)
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VGC_GAN(nn.Module):
    def __init__(self, num_stock, d_market, d_news, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2,
                 t_mix=1):
        super(VGC_GAN, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        self.period = 73
        self.seq_len = 168
        self.seq_R = int(math.ceil(self.seq_len / self.period))
        time_step = 16
        scale_dim = time_step // 2

        if self.t_mix == 0:
            self.mixer_s = MultTime2dMixer(time_step, d_market + d_news, scale_dim=scale_dim)
            self.mixer_r = MultTime2dMixer(time_step, d_market + d_news, scale_dim=scale_dim)
        elif self.t_mix == 1:
            self.tensor = Graph_Tensor(num_stock, d_hidden, d_market, d_news)
            self.mixer_s = MultTime2dMixer(time_step, d_market, scale_dim=scale_dim)
            self.mixer_r = MultTime2dMixer(time_step, d_market, scale_dim=scale_dim)

        self.conv = nn.Conv1d(in_channels=d_market, out_channels=d_market, kernel_size=2, stride=2)
        self.channel_fc = nn.Linear(d_market, 1)
        self.graph_generator = Graph_Generator(nnodes=num_stock, k=5, dim=hidn_rnn,
                                               device='cuda' if torch.cuda.is_available() else 'cpu')
        self.attentions = [
            Graph_Attention_v2(hidn_rnn, hidn_att, dropout=dropout, alpha=alpha, residual=True, num_heads=heads_att) for
            _ in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.hypergat = HyperGAT(hidn_rnn, hidn_att, output_size=hidn_att, beta=0.5, dropout=dropout)
        self.Linearx = nn.Linear(heads_att * hidn_att + hidn_rnn + hidn_att, heads_att * hidn_att + hidn_rnn)
        self.X2Os = Graph_Linear(num_stock, heads_att * hidn_att + hidn_rnn, 2, bias=True)
        self.linear_mixer = nn.Linear(40, 78)
        self.generator = nn.Sequential(
            nn.Linear(heads_att * hidn_att + hidn_rnn, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, heads_att * hidn_att + hidn_rnn)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(heads_att * hidn_att + hidn_rnn, d_hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        )
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, x_market, x_news, relation_static=None, adversarial=False):
        x_market = x_market.float()
        x_news = x_news.float()

        if self.t_mix == 0:
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:
            x_s = self.tensor(x_market, x_news)
            x_r = self.tensor(x_market, x_news)

        x_s = x_s.permute(1, 0, 2)
        x_s1 = x_s.permute(0, 2, 1)
        x_s1 = self.conv(x_s1)
        x_s1 = x_s1.permute(0, 2, 1)
        x_s = self.mixer_s(x_s, x_s1)
        x_s = self.channel_fc(x_s).squeeze(-1)

        x_r = x_r.permute(1, 0, 2)
        x_r1 = x_r.permute(0, 2, 1)
        x_r1 = self.conv(x_r1)
        x_r1 = x_r1.permute(0, 2, 1)
        x_r = self.mixer_s(x_r, x_r1)
        x_r = self.channel_fc(x_r).squeeze(-1)

        x_s = self.linear_mixer(x_s)
        x_r = self.linear_mixer(x_r)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        x_s = F.dropout(x_s, self.dropout, training=self.training)

        adj_imp = self.graph_generator(torch.arange(x_r.size(0)).to(x_r.device))
        x_attention = torch.cat([att(x_s, x_r, relation_static=adj_imp) for att in self.attentions], dim=1)
        x_attention = F.dropout(x_attention, self.dropout, training=self.training)
        hyper_x = self.hypergat(x_r, relation_static)
        hyper_x = F.dropout(hyper_x, self.dropout, training=self.training)
        x = torch.cat([x_s, x_attention, hyper_x], dim=1)
        x = self.Linearx(x)

        if adversarial:
            fake_features = self.generator(x.detach())
            fake_score = self.discriminator(fake_features)
            real_score = self.discriminator(x)
            return fake_score, real_score

        x = F.elu(self.X2Os(x))
        output = F.log_softmax(x, dim=1)
        return output


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch*nodes, feat, time)
        return self.act(self.norm(self.conv(x)))


class TimeMixer(nn.Module):
    def __init__(self, seq_len, feat_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = feat_dim
        self.tcb1 = TemporalConvBlock(feat_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tcb2 = TemporalConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: (batch, nodes, seq, feat)
        b, n, seq, f = x.shape
        x = x.view(b * n, seq, f).permute(0, 2, 1)  # (b*n, feat, seq)
        h = self.tcb1(x)
        h = self.tcb2(h)
        h = self.pool(h).squeeze(-1)
        h = self.fc(h)
        h = h.view(b, n, -1)
        return h  # (batch, nodes, hidden)


class ChannelFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_market, x_news):
        # x_market/x_news: (batch, nodes, feat)
        x = torch.cat([x_market, x_news], dim=-1)
        return self.fc2(self.act(self.fc1(x)))  # (batch, nodes, hidden)


class GraphGenerator(nn.Module):
    def __init__(self, nnodes, hidden_dim=64):
        super().__init__()
        self.n = nnodes
        self.emb = nn.Parameter(torch.randn(nnodes, hidden_dim) * 0.1)
        self.affine = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, device=None):
        A = torch.matmul(self.emb, self.emb.t())
        A = torch.tanh(A)
        A = (A + A.t()) / 2.0
        A = torch.sigmoid(A)
        if device is not None:
            return A.to(device)
        return A


class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(heads)])
        self.a_src = nn.ParameterList([nn.Parameter(torch.Tensor(out_dim)) for _ in range(heads)])
        self.a_dst = nn.ParameterList([nn.Parameter(torch.Tensor(out_dim)) for _ in range(heads)])
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, A=None):
        # x: (batch, nodes, feat)
        b, n, f = x.shape
        outs = []
        for i in range(self.heads):
            h = self.linears[i](x)  # (b, n, out_dim)
            a1 = (h * self.a_src[i]).sum(-1, keepdim=True)
            a2 = (h * self.a_dst[i]).sum(-1, keepdim=True)
            att = a1 + a2.permute(0, 2, 1)  # (b, n, n)
            att = self.leaky(att)
            if A is not None:
                if A.dim() == 2:
                    A_ = A.unsqueeze(0).expand(b, -1, -1)
                else:
                    A_ = A
                att = att + (A_ * 5.0)
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            out = torch.bmm(att, h)
            outs.append(out)
        h_cat = torch.cat(outs, dim=-1)
        return F.elu(h_cat)


class Readout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, task='class'):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        if task == 'class':
            self.fc2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, 1)
        self.task = task

    def forward(self, x):
        # x: (batch, nodes, feat)
        h = x.mean(dim=1)
        h = self.act(self.fc1(h))
        out = self.fc2(h)
        if self.task == 'class':
            return F.log_softmax(out, dim=-1)
        else:
            return out


class MagicNet(nn.Module):
    def __init__(self,
                 num_stock,
                 d_market,
                 d_news,
                 seq_len,
                 hidden_time=64,
                 hidden_channel=128,
                 gat_heads=4,
                 gat_out_per_head=32,
                 readout_hidden=128,
                 num_classes=2,
                 dropout=0.1,
                 use_relation=True):
        super().__init__()
        self.num_stock = num_stock
        self.seq_len = seq_len
        self.use_relation = use_relation
        self.time_market = TimeMixer(seq_len, d_market, hidden_dim=hidden_time)
        self.time_news = TimeMixer(seq_len, d_news, hidden_dim=hidden_time)
        self.channel_fuse = ChannelFusion(hidden_time * 2, hidden_channel)
        self.channel_proj = nn.Linear(hidden_channel, gat_out_per_head * gat_heads)
        self.graph_gen = GraphGenerator(nnodes=num_stock, hidden_dim=hidden_channel)
        self.gat = SimpleGATLayer(in_dim=gat_out_per_head * gat_heads, out_dim=gat_out_per_head, heads=gat_heads,
                                  dropout=dropout)
        self.hyper = nn.Sequential(
            nn.Linear(gat_out_per_head * gat_heads, gat_out_per_head * gat_heads),
            nn.GELU(),
            nn.Linear(gat_out_per_head * gat_heads, gat_out_per_head * gat_heads)
        )
        self.readout = Readout(in_dim=gat_out_per_head * gat_heads, hidden_dim=readout_hidden, out_dim=num_classes,
                               task='class' if num_classes > 1 else 'reg')
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_market, x_news, relation_static=None):
        # x_market, x_news: (batch, nodes, seq_len, feat)
        x_market = x_market.float()
        x_news = x_news.float()
        b, n, seq, dm = x_market.shape
        tm = self.time_market(x_market)
        tn = self.time_news(x_news)
        fused = self.channel_fuse(tm, tn)  # (b, n, hidden_channel)
        proj = self.channel_proj(fused)  # (b, n, gat_dim)
        proj = F.dropout(proj, self.dropout.p, training=self.training) if isinstance(self.dropout, nn.Dropout) else proj
        if relation_static is None and self.use_relation:
            A = self.graph_gen(device=proj.device)
        else:
            A = relation_static
        gat_h = self.gat(proj, A)  # (b, n, out_dim*heads)
        hyper_h = self.hyper(gat_h)
        h = torch.cat([gat_h, hyper_h], dim=-1)
        h = F.dropout(h, p=0.1, training=self.training)
        out = self.readout(h)
        return out


if __name__ == "__main__":
    batch = 4
    num_stock = 25
    seq_len = 73
    d_market = 16
    d_news = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MagicNet(num_stock=num_stock,
                   d_market=d_market,
                   d_news=d_news,
                   seq_len=seq_len,
                   hidden_time=64,
                   hidden_channel=128,
                   gat_heads=4,
                   gat_out_per_head=32,
                   readout_hidden=128,
                   num_classes=2,
                   dropout=0.1).to(device)
    xm = torch.randn(batch, num_stock, seq_len, d_market).to(device)
    xn = torch.randn(batch, num_stock, seq_len, d_news).to(device)
    logits = net(xm, xn)
    print("输出形状:", logits.shape)
