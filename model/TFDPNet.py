import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import torch.nn.functional as F
from model.RevIN import RevIN
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.batch = configs.batch_size
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len
        self.multivariate = configs.enc_in
        self.use_norm = configs.use_norm
        self.output_attention = configs.output_attention
        self.d_model_frequency = configs.d_model_frequency
        self.top_p = configs.top_p

        self.num_experts = self.multivariate
        self.heads = configs.n_heads
        self.alpha = configs.alpha
        ### Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)

        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)

        revin = True
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(self.multivariate, affine=True, subtract_last=False)

        self.freq_dim = int((self.seq_len / 2) + 1)

        mask_flag = True
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(mask_flag, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_model*2,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )


        self.fc_2 = nn.Linear(self.seq_len, self.d_model)
        self.fc_3 = nn.Linear(self.d_model, self.pred_len )
        self.fc_4 = nn.Linear(self.seq_len, self.d_model)

        ## 时间聚类
        self.fc_time_clean = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_model//2, self.num_experts, bias=False)
        )
        self.fc_time_noisy = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_model//2, self.num_experts, bias=False)
        )

        # self.softplus = F.sigmoid()
        self.noise_epsilon = 1e-2
        self.softplus = nn.Softplus(-1)
        # self.W_h = nn.Parameter(torch.eye(self.num_experts))

        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.LeakyReLU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.y_freq = nn.Linear((math.ceil(self.freq_dim * self.alpha )) * 2 , self.d_model_frequency)

        self.linear_sean = nn.Sequential(
            nn.Linear(self.d_model_frequency, self.d_model_frequency),
            nn.LeakyReLU(),
            nn.Linear(self.d_model_frequency, self.d_model_frequency)
        )

        self.fc_out = nn.Linear(self.d_model*2, self.pred_len)

        ##################
        self.key_frequency_len = configs.key_frequency
        if self.multivariate == 862:
            self.data = torch.nn.Parameter(torch.randn(self.key_frequency_len, self.multivariate))
        else:
            self.data = torch.nn.Parameter(torch.zeros(self.key_frequency_len, self.multivariate))

        self.fc_5 = nn.Linear(self.d_model, self.pred_len)


    def Time_Clustering(self, x):
        # bath variate d_model
        x_clean = self.dropout(self.fc_time_clean(x))
        x_noisy = self.dropout(self.fc_time_noisy(x))
        x_experts = x_clean + torch.randn(x_clean.size(-1), device=x_clean.device) * self.softplus(x_noisy + self.noise_epsilon)
        # x_experts = x_experts * self.W_h    # batch variate num_experts
        # logits = self.softmax(x_experts)
        x_min = x_experts.min(dim=-1, keepdim=True)[0]
        x_max = x_experts.max(dim=-1, keepdim=True)[0]
        logits = (x_experts - x_min) / (x_max - x_min + 1e-6)  # 避免除零


        identity_matrix = torch.eye(self.num_experts).cuda()
        logits = logits + identity_matrix
        mask = logits < self.top_p

        return mask.unsqueeze(1)#.repeat(1,self.heads,1,1)


    def freq_filter(self,y_real, y_imag ):

        length = math.ceil(self.freq_dim * self.alpha)
        y_freq_key = torch.cat([y_real[:,: , :length], y_imag[:,: , :length] ], dim=-1)   #  batch variate 98
        y_freq_noise = torch.cat([y_real[:,: , length:], y_imag[:,: , length:] ], dim=-1)          #  batch variate 98

        y_freq_complete = torch.cat([y_real, y_imag], dim=-1)

        return y_freq_key, y_freq_noise, y_freq_complete

    def generate_shared_mask(seelf,B, L , N):
        # 初始化全False的掩码矩阵
        mask_ratio = 0.1
        mask = torch.zeros((N, N), dtype=torch.bool)

        # 生成随机掩码索引
        total_elements = N * N  # 总元素数
        num_masked = int(mask_ratio * (total_elements - N))  # 计算10%的非对角线元素

        # 生成 num_vars * num_vars 的随机索引
        indices = torch.tril_indices(N, N, offset=-1)  # 获取下三角索引（不含对角线）
        perm = torch.randperm(indices.shape[1])[:num_masked]  # 随机选择部分索引
        selected_indices = indices[:, perm]  # 选出的索引

        # 赋值掩码（对称地设置上下三角）
        mask[selected_indices[0], selected_indices[1]] = True
        mask[selected_indices[1], selected_indices[0]] = True

        # 扩展 batch 维度，使所有 batch 共享相同的掩码
        mask = mask.unsqueeze(0).expand(B, -1, -1)  # (batch_size, num_vars, num_vars)

        return mask

    def key_frequency_learning(self,index, length):

        gather_index = index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)
        gather_index = gather_index % self.key_frequency_len
        Shared_frequency = self.data[gather_index]
        return Shared_frequency


    def Frequency_encoder(self, x_enc, key_frequency_index, B, L, N):
        seasonal_init, trend_init = self.decompsition(x_enc)
        trend_init = self.dropout(self.linear_trend(trend_init.permute(0, 2, 1)) )

        x_enc_fft = torch.fft.rfft(x_enc.transpose(-1, -2), dim=-1, norm='ortho')  # B N L/2 +1
        y_real, y_imag = x_enc_fft.real, x_enc_fft.imag

        y_freq_key, y_freq_noise, y_freq_complete = self.freq_filter(y_real, y_imag)
        # random_mask = self.generate_shared_mask(B, L, N).cuda()

        y_freq_key = self.dropout( self.y_freq(y_freq_key) )

        y_freq_key = y_freq_key.transpose(2, 1) - self.key_frequency_learning(key_frequency_index,y_freq_key.size(-1))  # batch length variate

        y_freq = self.dropout(self.linear_sean(y_freq_key.transpose(2, 1)))
        # y_freq = self.dropout(self.linear_sean(y_freq_key))

        y_freq = y_freq.transpose(2, 1) + self.key_frequency_learning((key_frequency_index + y_freq.size(-1)) % self.key_frequency_len, y_freq.size(-1))
        y_freq = y_freq.transpose(2, 1)

        x_fre = torch.fft.irfft(y_freq, n=L, dim=-1, norm='ortho')  # batch variate seq_len
        x_fre = self.fc_4(x_fre) + trend_init

        x_fre_loss = self.fc_5(x_fre)

        return x_fre, x_fre_loss

    def Time_encoder(self,x_enc):
        #### batch length variate
        x_bedding = self.fc_2(x_enc.transpose(-1, -2))
        attn_mask = self.Time_Clustering(x_bedding)
        # attn_mask = None
        x_bedding, _ = self.encoder(x_bedding, attn_mask)
        x_time_loss = self.fc_3(x_bedding)   # 可有可无

        return x_bedding, x_time_loss

    def forecast(self, x_enc, key_frequency_index):

        if self.use_norm:
            x_enc = self.revin_layer(x_enc, 'norm')
            # mean_enc = x_enc.mean(1, keepdim=True).detach()
            # x_enc = x_enc - mean_enc
            # std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            # x_enc = x_enc / std_enc

        B, L, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariate

        # ## frequency-dimentional encoder
        x_fre, x_fre_loss = self.Frequency_encoder(x_enc, key_frequency_index, B, L, N)

        # ## time-dimentional encoder
        x_time, x_time_loss = self.Time_encoder(x_enc)       # batch variate

        x = torch.cat([x_time,x_fre],dim=-1)
        x = self.fc_out(x)

        dec_out = x.transpose(2,1)


        # # # # # # # # #
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = self.revin_layer(dec_out, 'denorm')
            x_time_loss = self.revin_layer(x_time_loss.transpose(2,1), 'denorm')
            x_fre_loss = self.revin_layer(x_fre_loss.transpose(2,1), 'denorm')
        else:
            dec_out = dec_out
            x_time_loss = x_time_loss.transpose(2, 1)
            x_fre_loss = x_fre_loss.transpose(2, 1)

        return dec_out, x_time_loss, x_fre_loss

    def forward(self, x_enc, batch_key_frequency, mask=None):
        dec_out, x_time_loss, x_fre_loss = self.forecast(x_enc, batch_key_frequency)

        return dec_out[:, -self.pred_len:, :], x_time_loss, x_fre_loss   # [B, L, D]


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
