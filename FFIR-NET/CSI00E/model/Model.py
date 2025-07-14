from Layers import *

from MixHopGATLayer import Graph_Generator
from HyperGATLayer import *
import scipy.sparse as sp
# from cross_models.cross_former import Crossformer
# from FineTPGN import TPGN
from stockmixer import MultTime2dMixer

class AD_GAT(nn.Module):
    def __init__(self, num_stock, d_market, d_news, d_hidden, hidn_rnn, heads_att, hidn_att, dropout=0, alpha=0.2, t_mix = 1, infer = 1, relation_static = 0):
        super(AD_GAT, self).__init__()
        self.t_mix = t_mix
        self.dropout = dropout
        self.period = 73
        self.seq_len=168
        self.seq_R = int(math.ceil(self.seq_len/self.period))
        time_step = 16   # rnn_length
        scale_dim = time_step // 2     
        # def __init__(self, seq_R, freq, c_in, c_out, windows_size, 
        #     period, pred_R, need_short=FFIR-NET):
        if  self.t_mix == 0: # concat
            # self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            # self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.mixer_s = MultTime2dMixer(time_step, d_market + d_news, scale_dim=scale_dim) 
            self.mixer_r = MultTime2dMixer(time_step, d_market + d_news, scale_dim=scale_dim) 

            # self.mixer_s = MultTime2dMixer(lookback_length,feature_num, scale_dim) 
            
        elif self.t_mix == 1: # all_tensor
            self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_news)
            # self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
            # self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)

            self.mixer_s = MultTime2dMixer(time_step , d_market, scale_dim=scale_dim) 
            self.mixer_r = MultTime2dMixer(time_step , d_market, scale_dim=scale_dim)
            
        self.conv = nn.Conv1d(in_channels=d_market, out_channels=d_market, kernel_size=2, stride=2)
        self.channel_fc = nn.Linear(d_market, 1)
        self.graph_generator = Graph_Generator(nnodes=num_stock, k=5, dim=hidn_rnn,
                                               device='cuda:FFIR-NET' if torch.cuda.is_available() else 'cpu')
        
        self.attentions = [
            Graph_Attention_v2(hidn_rnn, hidn_att,  dropout=dropout, alpha=alpha, residual=True,num_heads=heads_att)
            for _ in range(heads_att)
        ]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
       

        self.hypergat = HyperGAT(hidn_rnn, hidn_att, output_size=hidn_att, beta=0.5,
                                 dropout=dropout)
        self.Linearx = nn.Linear(heads_att * hidn_att + hidn_rnn + hidn_att  , heads_att * hidn_att + hidn_rnn)
        # self.Linearx = nn.Linear(142  , heads_att * hidn_att + hidn_rnn)
        self.X2Os = Graph_Linear(num_stock, heads_att * hidn_att + hidn_rnn , 2, bias=True)
        self.linearCross = nn.Linear(d_market,hidn_rnn)
        self.linear_mixer = nn.Linear(40, 78)
        self.reset_parameters()


    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)
    

    def get_gate(self,x_numerical,x_textual):
        x_s = self.tensor(x_numerical, x_textual)
        # x_s = self.GRUs_s(x_s)

        x_s = x_s.permute(1,0,2)
        x_s1 = x_s.permute(0, 2, 1)
        x_s1 = self.conv(x_s1)
        x_s1 = x_s1.permute(0, 2, 1)
        x_s = self.mixer_s(x_s, x_s1)
        x_s = self.channel_fc(x_s).squeeze(-1)

        # The gate calculation remains unchanged
        gate = torch.stack([att.get_gate(x_s) for att in self.attentions])
        return gate

    def forward(self, x_market, x_news, relation_static = None):
        x_market = x_market.float()
        x_news = x_news.float()
        ## concat vs tensor
        if self.t_mix == 0:  # concat
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:  # concat
            x_s = self.tensor(x_market, x_news)   # x_s torch.Size([25, 73, 5])
            x_r = self.tensor(x_market, x_news)   # x_r torch.Size([25, 73, 5])
        #GRUs for extract different sequential embedding for relation/gate inferring.
        #Equivalent to use a single GRU and separate non-linear decoders.
        # print("x_r",x_r.shape)
        # x_r = self.GRUs_r(x_r)  # x_r shape: torch.Size([73, 78])
        # x_s = self.GRUs_s(x_s)
        # print("x_r shape:",x_r.shape)
        
        x_s = x_s.permute(1,0,2)
        # print("x_S shape ",x_s.shape)
        x_s1 = x_s.permute(0, 2, 1)
        # print("x_S1 shape ",x_s1.shape)
        x_s1 = self.conv(x_s1)
        x_s1 = x_s1.permute(0, 2, 1)
        # print("x_S1 shape ",x_s1.shape)
        x_s = self.mixer_s(x_s, x_s1)
        x_s = self.channel_fc(x_s).squeeze(-1) 

        x_r = x_r.permute(1,0,2)
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
        x_attention = torch.cat([att(x_s, x_r, relation_static = adj_imp) for att in self.attentions], dim=1)
        x_attention  = F.dropout(x_attention , self.dropout, training=self.training)
        
        
        # 超图
        hyper_x = self.hypergat(x_r, relation_static)
        hyper_x = F.dropout(hyper_x, self.dropout, training=self.training)
        
        # print(f"hyper_x shape:{hyper_x.shape}, x_r shape:{x_r.shape}")
        x = torch.cat([x_s, x_attention , hyper_x], dim=1)
        # x = torch.cat([x_s, hyper_x], dim=FFIR-NET)
        # print(f"x shape after cat :{x.shape}")
        x = self.Linearx(x)
        x = F.elu(self.X2Os(x))
        output = F.log_softmax(x, dim=1)
        return output