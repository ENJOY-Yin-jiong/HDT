import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .operation import Conv1D


class TemporalAttentionModule(nn.Module):
    def __init__(self, dim, num_heads, scale=3, linear_padding=1, drop_rate=0.0):
        super(TemporalAttentionModule, self).__init__()
        self.dropout = nn.Dropout(p=drop_rate)
        self.scale = scale
        self.linear_padding = linear_padding
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim

        self.tem_query = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.tem_key = Conv1D(in_dim=dim,
                                out_dim=dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)
        self.tem_value = Conv1D(in_dim=dim,
                                out_dim=dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)

        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

        self.fwd_bkw_layer = Conv1D(in_dim=2*dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)


        self.output_activation = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)  # (batch_size, w_seq_len, num_heads, scales, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def dislocation_and_padding(self, input_feature, reverse=False):
        _, T, _ = input_feature.shape
        # Linear
        intermediate = []
        for i in range(self.scale):
            if reverse:
                step_scale = F.pad(input_feature,
                                   pad=(
                                   0, 0, (self.scale - 1 - i) * self.linear_padding, i * self.linear_padding, 0, 0),
                                   mode='constant',
                                   value=0).unsqueeze(2)
            else:
                step_scale = F.pad(input_feature,
                                   pad=(0, 0, i * self.linear_padding, (self.scale - 1 - i) * self.linear_padding, 0, 0),
                                   mode='constant',
                                   value=0).unsqueeze(2)
            intermediate.append(step_scale)
        intermediate = torch.cat(intermediate, dim=2)  # batch_size, sequence_length, scale_size, dim
        if reverse:
            _, _, L, _ =intermediate.shape
            intermediate = intermediate[:, L-T-self.scale:, :, :]
        else:
            intermediate = intermediate[:, :T, :, :]
        return intermediate



    def forward(self, input_feature, query_feature=None):
        
        input_feature = self.layer_norm1(input_feature)

        query = self.tem_query(input_feature)
        key = self.tem_key(input_feature)
        value = self.tem_value(input_feature)

        query_fwd = self.dislocation_and_padding(query)
        key_fwd = self.dislocation_and_padding(key)
        value_fwd = self.dislocation_and_padding(value)

        query_reverse = self.dislocation_and_padding(query, reverse=True)
        key_reverse = self.dislocation_and_padding(key, reverse=True)
        value_reverse = self.dislocation_and_padding(value, reverse=True)


        step_q = self.transpose_for_scores(query_fwd)
        step_k = self.transpose_for_scores(key_fwd)
        step_v = self.transpose_for_scores(value_fwd)

        step_q_reverse = self.transpose_for_scores(query_reverse)
        step_k_reverse = self.transpose_for_scores(key_reverse)
        step_v_reverse = self.transpose_for_scores(value_reverse)


        step_attn = torch.matmul(step_q, step_k.transpose(-1, -2))
        step_attn_reverse = torch.matmul(step_q_reverse, step_k_reverse.transpose(-1, -2))

        # print('step_attn:', step_attn.shape)
        step_attn = step_attn / math.sqrt(self.head_size)
        step_attn = F.softmax(step_attn, dim=-1)
        step_attn = self.dropout(step_attn)
        step_v = torch.matmul(step_attn, step_v)
        step_v = self.combine_last_two_dim(step_v.permute(0, 1, 3, 2, 4))
        step_v = self.dropout(step_v)

        # reverse
        step_attn_reverse = step_attn_reverse / math.sqrt(self.head_size)
        step_attn_reverse = F.softmax(step_attn_reverse, dim=-1)
        step_attn_reverse = self.dropout(step_attn_reverse)
        step_v_reverse = torch.matmul(step_attn_reverse, step_v_reverse)
        step_v_reverse = self.combine_last_two_dim(step_v_reverse.permute(0, 1, 3, 2, 4))
        step_v_reverse = self.dropout(step_v_reverse)

        out = step_v[:, :, 1, :]
        out_reverse = step_v_reverse[:, :, 1, :]

        out = torch.cat((out, out_reverse), dim=2)
        out = self.fwd_bkw_layer(out)

        out = out + input_feature
        out = self.layer_norm2(out)
        out = self.out_layer1(out)
        out = self.output_activation(out)
        out = self.dropout(out)

        return out


class TemporalAttentionEncoder(nn.Module):
    def __init__(self, dim, num_heads, scale=3, linear_padding=1, num_layers=1, drop_rate=0.0):
        super(TemporalAttentionEncoder, self).__init__()
        self.dropout = nn.Dropout(p=drop_rate)

        self.temporal_attention_layers = nn.ModuleList([
                                        TemporalAttentionModule(dim=dim,
                                                                num_heads=num_heads,
                                                                scale=scale+1,
                                                                linear_padding=i, # linear_padding=i+1,
                                                                drop_rate=drop_rate) for i in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

        self.out_layer = Conv1D(in_dim=dim,
                                out_dim=dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)

    def forward(self, input_feature, query_feature=None):
        out = input_feature
        for i, module in enumerate(self.temporal_attention_layers):
            out = self.layer_norm(out)
            out = module(out, query_feature)
        out = self.out_layer(out)
        return out
    

class MultiStepLSTMEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 num_layers,
                 num_step=1,
                 bi_direction=False,
                 drop_rate=0.0):
        super(MultiStepLSTMEncoder, self).__init__()

        self.num_step = num_step
        self.out_dim = out_dim
        self.layers_norm = nn.LayerNorm(in_dim, eps=1e-6)

        self.dropout = nn.Dropout(p=drop_rate)

        self.encoder = nn.ModuleList([
            nn.LSTM(in_dim,
                    out_dim // 2 if bi_direction else out_dim,
                    num_layers=num_layers,
                    bidirectional=bi_direction,
                    dropout=drop_rate,
                    batch_first=True) for _ in range(num_step)
        ])
        self.linear = Conv1D(in_dim=int(num_step * out_dim),
                             out_dim=out_dim,
                             kernel_size=1,
                             stride=1,
                             bias=True,
                             padding=0)

    def forward(self, input_feature):
        input_feature = self.layers_norm(input_feature)
        B, seq_len, _ = input_feature.shape
        # assert seq_len // self.num_step == 0, "length of sequence({}) must be devided by num_step({})".format(
        #     seq_len, self.num_step)
        output = []
        for i in range(self.num_step):
            encoder_i = self.encoder[i]
            output_i = input_feature.new_zeros([B, seq_len, self.out_dim])
            input_i_len = (seq_len // (i + 1)) * (i + 1)
            for j in range(i + 1):
                input_j = input_feature[:, j:input_i_len:(i + 1), :]
                output_j, _ = encoder_i(input_j)
                output_i[:, j:input_i_len:(i + 1), :] = output_j
            output_i = self.dropout(output_i)
            output.append(output_i)
        output = torch.cat(output, dim=2)
        output = self.linear(output)
        return output
    


class TemporalContextModule(nn.Module):
    def __init__(self, in_dim, out_dim, kernels=[3], drop_rate=0.):
        super(TemporalContextModule, self).__init__()
        self.dropout = nn.Dropout(p=drop_rate)
        self.temporal_convs = nn.ModuleList([
            Conv1D(in_dim=in_dim,
                   out_dim=out_dim,
                   kernel_size=s,
                   stride=1,
                   padding=s // 2,
                   bias=True) for s in kernels
        ])
        self.out_layer = Conv1D(in_dim=out_dim * len(kernels),
                                out_dim=out_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=True)

    def forward(self, input_feature):
        intermediate = []
        for layer in self.temporal_convs:
            intermediate.append(layer(input_feature))
        intermediate = torch.cat(intermediate, dim=-1)
        out = self.out_layer(intermediate)
        return out
