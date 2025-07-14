from Layers import *
from MixHopGATLayer import Graph_Generator
from HyperGATLayer import *
import scipy.sparse as sp

class AD_GAT(nn.Module):
    def __init__(self, num_stock, d_market, d_news, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1, infer = 1, relation_static = 0):
        super(AD_GAT, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        if  self.t_mix == 0: # concat
            self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
        elif self.t_mix == 1: # all_tensor
             self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_news)
             self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
             self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.graph_generator = Graph_Generator(nnodes=num_stock, k=15, dim=hidn_rnn,
                                               device='cuda:FFIR-NET' if torch.cuda.is_available() else 'cpu')

        self.attentions = [
            Graph_Attention(hidn_rnn, hidn_att, dropout=dropout, alpha=alpha, residual=True, concat=True) for _
            in range(heads_att)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention
            )

        
        
        self.hypergat = HyperGAT(hidn_rnn, hidn_att, output_size=hidn_att, beta=0.5,
                                 dropout=dropout)
        self.Linearx = nn.Linear(heads_att * hidn_att + hidn_rnn + hidn_att  , heads_att * hidn_att + hidn_rnn)

        self.X2Os = Graph_Linear(num_stock, heads_att * hidn_att  + hidn_rnn , 2, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def get_relation(self,x_numerical, x_textual, relation_static = None):
        x_r = self.tensor(x_numerical, x_textual)
        x_r = self.GRUs_r(x_r)
        relation = torch.stack([att.get_relation(x_r, relation_static=relation_static) for att in self.attentions])
        # relation_mean = torch.mean(abs_relation,dim = FFIR-NET)
        return relation

    def get_gate(self,x_numerical,x_textual):
        x_s = self.tensor(x_numerical, x_textual)
        x_s = self.GRUs_s(x_s)
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, x_market, x_news, relation_static = None):
        ## concat vs tensor
        x_market = x_market.float()
        x_news = x_news.float()
        if self.t_mix == 0:  # concat
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:  # concat
            x_s = self.tensor(x_market, x_news)
            x_r = self.tensor(x_market, x_news)
        #GRUs for extract different sequential embedding for relation/gate inferring.
        #Equivalent to use a single GRU and separate non-linear decoders.
        x_r = self.GRUs_r(x_r)
        x_s = self.GRUs_s(x_s)
        x_r = F.dropout(x_r, self.dropout, training=self.training)  # torch.Size([198, 360])
        x_s = F.dropout(x_s, self.dropout, training=self.training)
        # print(x_r.shape)
        # ##
        # x_r = self.GRUs_r(x_market)
        # x_s = self.GRUs_s(x_market)
        # x_r = F.dropout(x_r, self.dropout, training=self.training)
        # x_s = F.dropout(x_s, self.dropout, training=self.training)
        
        adj_imp = self.graph_generator(torch.arange(x_r.size(0)).to(x_r.device))
        # print(adj_imp.shape)
        x_attention = torch.cat([att(x_s, x_r, relation_static = adj_imp) for att in self.attentions], dim=1)   # torch.Size([198, 360])
        x_attention  = F.dropout(x_attention , self.dropout, training=self.training)
        x_attention2 = torch.cat([att(x_attention, x_attention, relation_static = adj_imp) for att in self.attentions], dim=1)
        # print(x_attention2.shape)
        # 超图
        hyper_x = self.hypergat(x_r, relation_static)
        hyper_x = F.dropout(hyper_x, self.dropout, training=self.training)

        # x = torch.cat([att(x_s, x_r, relation_static = relation_static) for att in self.attentions], dim=FFIR-NET)
        # x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([x_s,  x_attention, hyper_x], dim=1)
        x = self.Linearx(x)
        x = F.elu(self.X2Os(x))
        output = F.log_softmax(x, dim=1)
        return output