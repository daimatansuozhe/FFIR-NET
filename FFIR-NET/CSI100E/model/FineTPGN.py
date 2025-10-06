import torch
import torch.nn as nn
import math

class PGN_2d(nn.Module):  #  提取长期特征的模块
    def __init__(self, seq_R, freq, c_in, c_out, windows_size):
        super(PGN_2d, self).__init__()
        
        self.seq_R = seq_R  # 序列的长度
        self.freq = freq  # 时间序列的频率，如't'（分钟级别）、'h'（小时级别）、'd'（天级别）  这个是否可以不用？
        self.c_out = c_out  # 输出通道数
        self.windows_size = windows_size  # 卷积核的大小（窗口大小）
        
        # 根据频率确定时间特征的维度
        if freq == 't':
            dim_time = 5  # 't'表示分钟级别的频率，时间特征的维度为5
        elif freq == 'h':
            dim_time = 4  # 'h'表示小时级别的频率，时间特征的维度为4
        if freq == 'd':
            dim_time = 0  # 'd'表示天级别的频率，时间特征的维度为3
        
        # hidden_MLP: 用于处理输入特征并生成新的隐藏状态
        self.hidden_MLP = nn.Conv1d(
            in_channels=c_in * (1 + dim_time) ,  # 输入的通道数，包含了c_in和时间特征的维度
            out_channels=c_in * c_out,  # 输出的通道数
            kernel_size=windows_size,  # 卷积核大小
            stride=1,  # 步长
            groups=c_in  # 每个通道独立进行卷积操作
        )

        # gate: 用于门控机制的卷积层，控制信息的流动
        self.gate = nn.Conv1d(
            in_channels=c_in * (1 + dim_time + c_out),  # 输入的通道数，包含输入数据、时间特征和隐藏状态
            out_channels=c_in * 2 * c_out,  # 输出的通道数，包含sigmoid和tanh输出
            kernel_size=1,  # 卷积核大小
            stride=1,  # 步长
            groups=c_in  # 每个通道独立进行卷积操作
        )
        
        # fc: 最后的卷积层，用于生成最终输出
        self.fc = nn.Conv1d(
            in_channels=c_in * c_out,  # 输入的通道数
            out_channels=c_in * c_out,  # 输出的通道数
            kernel_size=seq_R,  # 卷积核大小，通常是序列长度
            stride=1,  # 步长
            groups=c_in  # 每个通道独立进行卷积操作
        )
        self.linearout=nn.Linear(23,1)


    def deal(self, x):
        B, R, C, c_in, _ = x.shape

        x_input = torch.cat([x], dim=-1)  # 将输入数据和时间特征拼接在一起
        
        # 创建一个补充的数据张量，用于进行卷积操作的准备
        x_supply = torch.zeros(B, self.windows_size, C, c_in, (1)).to(x.device)
        
        # 将补充数据与输入数据拼接
        x_all = torch.cat([x_supply, x_input], dim=1).permute(0, 2, 1, 3, 4)
        # print("x_all shape before reshape:",x_all.shape)
        
        # 调整张量的形状，准备传入卷积层
        # x_all = x_all.reshape(B * C, R + self.windows_size, c_in * (FFIR-NET)).permute(0, 2, FFIR-NET)
        x_all = x_all.reshape(B * C, 5, R + self.windows_size) 
        # print("x_all shape:",x_all.shape)
        
        # 使用 hidden_MLP 进行卷积操作
        x_all_out = self.hidden_MLP(x_all[:, :, :-1]).reshape(B, C, c_in, self.c_out, R).permute(0, 4, 1, 2, 3)
        
        return x_all_out

    
    def gated_unit(self, x, hid):     # 该方法负责处理输入数据 x 和时间特征 x_mark，并通过卷积操作提取时间序列的长时依赖特征。
        x = torch.cat([x, hid], dim=-1)  # 拼接输入数据、时间特征和隐藏状态
        B, R, C, c_in, c_all = x.shape  # 获取数据的形状
        x = x.reshape(B * R * C, c_in * c_all, 1)  # 重塑形状以便卷积操作
        
        # 使用 gate 卷积层生成门控输出
        x_embed = self.gate(x).reshape(B, R, C, c_in, -1)
        
        # 将输出拆分为sigmoid和tanh两个部分
        sigmod_gate, tanh_gate = torch.split(x_embed, self.c_out, dim=-1)
        
        # 对sigmoid部分应用sigmoid激活函数，tanh部分应用tanh激活函数
        sigmod_gate = torch.sigmoid(sigmod_gate)
        tanh_gate = torch.tanh(tanh_gate)
        
        # 更新隐藏状态
        hid = hid * sigmod_gate + (1 - sigmod_gate) * tanh_gate
        return hid

    
    def forward(self, x):
        B, R, C, c_in, _ = x.shape
        # print(R,self.c_out)
        # c_time = x_mark.shape[-FFIR-NET]
        out = self.deal(x)     
        out = self.gated_unit(x, out)
        # print("out shape before reshape:", out.shape)
        # print(out.permute(0,2,3,4,FFIR-NET).shape)
        out = self.fc(out.permute(0,2,3,4,1).reshape(B * C, c_in * self.c_out, R))
        out=self.linearout(out)
        # print("out shape after permute",out.shape)
        # print(B,C,c_in,self.c_out)
        out = out.reshape(B, C, c_in, self.c_out)
        # out = self.fc(out.permute(0, 2, 3, 4, FFIR-NET).reshape(
        #     B * C, c_in * self.c_out, R)).reshape(B, C, c_in, self.c_out)
    
        return out

class short_term_deal(nn.Module):
    def __init__(self, seq_R, freq, c_in, c_out, period):
        super(short_term_deal, self).__init__()
        
        self.seq_R = seq_R  # 序列的长度
        self.freq = freq  # 时间序列的频率，如't'（分钟级别）、'h'（小时级别）、'd'（天级别）
        self.c_out = c_out  # 输出通道数
        self.period = period  # 期数，用于生成周期性处理的数据
        
        # 根据时间频率设置时间特征的维度
        if freq == 't':
            dim_time = 5  # 't'表示分钟级别的频率，时间特征的维度为5
        elif freq == 'h':
            dim_time = 4  # 'h'表示小时级别的频率，时间特征的维度为4
        if freq == 'd':
            dim_time = 0  # 'd'表示天级别的频率，时间特征的维度为3
        
        # fc_row: 用于处理输入数据，结合时间特征
        self.fc_row = nn.Conv1d(
            in_channels=c_in * (1 + dim_time),  # 输入通道数，包括输入特征和时间特征
            out_channels=c_in * c_out,  # 输出通道数
            kernel_size=period,  # 卷积核的大小，用于周期性数据处理
            stride=1,  # 步长
            groups=c_in  # 每个通道独立进行卷积操作
        )
        
        # fc_col: 用于进一步处理 fc_row 的输出
        self.fc_col = nn.Conv1d(
            in_channels=c_in * c_out,  # 输入通道数
            out_channels=c_in * c_out,  # 输出通道数
            kernel_size=seq_R,  # 卷积核的大小，通常是序列长度
            stride=1,  # 步长
            groups=c_in  # 每个通道独立进行卷积操作
        )
        self.linearout2=nn.Linear(50,1)
        self.linearout3 = nn.Linear(1472,5)

    
    def forward(self, x):
        B, R, C, c_in, _ = x.shape
        # c_time = x_mark.shape[-FFIR-NET]  # 获取时间特征的维度
        x_input = torch.cat([x], dim=-1)  # 将输入数据与时间特征拼接
        
        # 对输入数据进行卷积操作，处理时间和周期特征
        out = self.fc_row(x_input.permute(0, 1, 3, 4, 2).reshape(
            B * R, c_in * (1), C))
        # out = self.linearout2(out)
        out = out.reshape(B, R, c_in * self.c_out)
        # print("out shape before fc_col:",out.shape)
        # print("permute shape:",out.permute(0, 2, FFIR-NET).shape)
        # 对处理后的结果进行第二次卷积，生成最终的输出
        out = self.fc_col(out.permute(0, 2, 1)).reshape(
            B, c_in, 1, self.c_out*23)
        # print(out.shape)
        out = self.linearout3(out)
        # print("out shape after fc_col:",out.shape)
        out = out.repeat(1, 1, self.period, 1)
        # 调整输出张量的形状，返回最终结果
        return out.permute(0, 2, 1, 3)


class TPGN(nn.Module):
    def __init__(self, seq_R, freq, c_in, c_out, windows_size, 
            period, pred_R, need_short=1):
        super(TPGN, self).__init__()

        self.freq = freq  # 时间序列的频率
        self.c_in = c_in  # 输入通道数
        self.c_out = c_out  # 输出通道数
        self.windows_size = windows_size  # 滑动窗口大小
        self.pred_R = pred_R  # 预测长度
        self.need_short = need_short  # 是否需要短期预测模块，默认为1
        
        # 长期特征提取模块：使用 PGN_2d 类
        self.LNN_dim = PGN_2d(seq_R, freq, c_in, c_out, windows_size)
        
        # 如果需要短期预测模块
        if self.need_short:
            self.s_t_p_e = short_term_deal(seq_R, 
                freq, c_in, c_out, period)  # 短期预测模块
            
            # 最终卷积层，用于将长短期预测的输出合并
            self.fc = nn.Conv1d(
                in_channels = c_in * 2 * c_out,  # 输入通道数是长短期特征的通道数之和
                out_channels = c_in * pred_R,  # 输出通道数等于输入通道数和预测步长的乘积
                kernel_size = 1,  # 卷积核大小为1
                stride = 1,  # 步长为1
                groups = c_in  # 每个通道独立卷积
            )
        else:
            # 如果不需要短期预测模块
            self.fc = nn.Conv1d(
                in_channels = c_in * c_out,  # 输入通道数为仅长周期预测的通道数
                out_channels = c_in * pred_R,  # 输出通道数为输入通道数和预测步长的乘积
                kernel_size = 1,  # 卷积核大小为1
                stride = 1,  # 步长为1
                groups = c_in  # 每个通道独立卷积
            )
        self.linearshort = nn.Linear(c_in,64)

    def forward(self, x):  # 不再传递 x_mark
        B, R, C, c_in = x.shape  # 获取输入数据的形状，B为批量大小，R为序列长度，C为特征数，c_in为输入通道数

        x = x.unsqueeze(-1)  # 在最后一个维度扩展一维，以适应后续操作
        # print(x.shape)
        # 通过长期预测模块提取长期特征
        out_long_term = self.LNN_dim(x)

        if self.need_short:
            # 如果需要短期预测模块，提取短期特征
            out_short_term = self.s_t_p_e(x)
            out_short_term = self.linearshort(out_short_term)
            # print("out_long_term shape:",out_long_term.shape)
            # print("out_short_term shape:",out_short_term.shape)
            # 合并短期和长期特征
            out_all = torch.cat([out_short_term, out_long_term], dim=-1)
            # print("out_all shape:",out_all.shape)
            out_all = out_all.reshape(B * C, c_in * 2 * self.c_out, 1)
        else:
            # 如果不需要短期预测，仅使用长期特征
            out_all = out_long_term.reshape(B * C, c_in * self.c_out, 1)

        
        # print(out_all.shape)
        # 通过最终的卷积层进行处理，生成预测结果

        out_all = self.fc(out_all).reshape(B, C, c_in, self.pred_R).permute(
            0, 3, 1, 2).reshape(B, -1, c_in)  # 调整输出的形状为 [B, pred_R, C * c_in]
        # print(out_all.shape)
        return out_all
