import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm
import torch.nn.functional as F

from .operation import Conv1D, mask_logits
from .encoder import TemporalAttentionEncoder, MultiStepLSTMEncoder, TemporalContextModule

class MultiHeadAttention(nn.Module):
    def __init__(self, configs):
        super(MultiHeadAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        self.key = Conv1D(in_dim=dim,
                          out_dim=dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.value = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output


class MultiTemporalAttention(nn.Module):
    def __init__(self, configs):
        super(MultiTemporalAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate

        scale = configs.MTA_scale
        num_layers = configs.MTA_num_layers

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)

        # self.mem = nn.Parameter(torch.randn([5, configs.dim]), requires_grad=True)
        # self.mem_back = nn.Parameter(torch.randn([5, configs.dim]), requires_grad=True)

        self.query = TemporalAttentionEncoder(dim=dim,
                                             num_heads=num_heads,
                                             scale=scale,
                                             linear_padding=1,
                                             num_layers=num_layers,
                                             drop_rate=drop_rate)
        self.key = TemporalAttentionEncoder(dim=dim,
                                             num_heads=num_heads,
                                             scale=scale,
                                             linear_padding=1,
                                             num_layers=num_layers,
                                             drop_rate=drop_rate)
        self.value = TemporalAttentionEncoder(dim=dim,
                                             num_heads=num_heads,
                                             scale=scale,
                                             linear_padding=1,
                                             num_layers=num_layers,
                                             drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Parameter)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, q=None, mask=None):
        # men = self.mem.unsqueeze(0).repeat(x.size(0), 1, 1)
        # men_back = self.mem_back.unsqueeze(0).repeat(x.size(0), 1, 1)
        # x = torch.cat((men, x), dim=1)
        # x = torch.cat((x, men), dim=1)

        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)

        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output, q))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output, q))
        value = self.transpose_for_scores(self.value(output, q))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        # output = output[:, 5:, :]
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # self.conv1 = Conv1D(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # self.conv2 = Conv1D(n_inputs, n_outputs, kernel_size=kernel_size, stride=stride, padding=padding,
        #                     dilation=dilation)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = x.transpose(1, 2)
        # print(x.shape)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(out + res)
        out = out.transpose(1, 2)
        return out


class TemporalDilationConv(nn.Module):
    def __init__(self, configs):
        # num_inputs, num_channels, kernel_size=2, dropout=0.2
        super(TemporalDilationConv, self).__init__()

        num_inputs =512
        num_channels = [512, 512, 512, 512]
        kernel_size = 4
        dropout = configs.drop_rate

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = i+1
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x, q=None, mask=None):
        out = self.network(x)
        return out


class MultiDilationConvAttention(nn.Module):
    def __init__(self, configs):
        super(MultiDilationConvAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate

        scale = configs.MTA_scale
        num_layers = configs.MTA_num_layers

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)

        self.query = TemporalDilationConv(configs)
        self.key = TemporalDilationConv(configs)
        self.value = TemporalDilationConv(configs)

        # self.query = TemporalAttentionEncoder(dim=dim,
        #                                      num_heads=num_heads,
        #                                      scale=scale,
        #                                      linear_padding=1,
        #                                      num_layers=num_layers,
        #                                      drop_rate=drop_rate)
        # self.key = TemporalAttentionEncoder(dim=dim,
        #                                      num_heads=num_heads,
        #                                      scale=scale,
        #                                      linear_padding=1,
        #                                      num_layers=num_layers,
        #                                      drop_rate=drop_rate)
        # self.value = TemporalAttentionEncoder(dim=dim,
        #                                      num_heads=num_heads,
        #                                      scale=scale,
        #                                      linear_padding=1,
        #                                      num_layers=num_layers,
        #                                      drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, q=None, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)

        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output, q))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output, q))
        value = self.transpose_for_scores(self.value(output, q))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output
    

# class PriorGuidedPromptAttention(nn.Module):
#     def __init__(self, configs):
#         super(PriorGuidedPromptAttention, self).__init__()
#         dim = configs.dim
#         num_heads = configs.num_heads
#         drop_rate = configs.drop_rate
#         assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
#             dim, num_heads)
#         self.head_size, self.num_heads, self.dim = int(
#             dim / num_heads), num_heads, dim
#         self.dropout = nn.Dropout(p=drop_rate)
#         self.query = Conv1D(in_dim=dim,
#                             out_dim=dim,
#                             kernel_size=1,
#                             stride=1,
#                             padding=0,
#                             bias=True)
#         self.key = Conv1D(in_dim=dim,
#                           out_dim=dim,
#                           kernel_size=1,
#                           stride=1,
#                           padding=0,
#                           bias=True)
#         self.value = Conv1D(in_dim=dim,
#                             out_dim=dim,
#                             kernel_size=1,
#                             stride=1,
#                             padding=0,
#                             bias=True)
#         # num_layers = 1
#         # num_step = 3
#         # bi_direction = True

#         # self.query = MultiStepLSTMEncoder(in_dim=dim,
#         #                                   out_dim=dim,
#         #                                   num_layers=num_layers,
#         #                                   num_step=num_step,
#         #                                   bi_direction=bi_direction,
#         #                                   drop_rate=drop_rate)
        
#         # self.key = MultiStepLSTMEncoder(in_dim=dim,
#         #                                 out_dim=dim,
#         #                                 num_layers=num_layers,
#         #                                 num_step=num_step,
#         #                                 bi_direction=bi_direction,
#         #                                 drop_rate=drop_rate)
#         # self.value = MultiStepLSTMEncoder(in_dim=dim,
#         #                                   out_dim=dim,
#         #                                   num_layers=num_layers,
#         #                                   num_step=num_step,
#         #                                   bi_direction=bi_direction,
#         #                                   drop_rate=drop_rate)

#         # self.value_visual = None
#         self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
#         self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
#         self.out_layer1 = Conv1D(in_dim=dim,
#                                  out_dim=dim,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0,
#                                  bias=True)
#         self.output_activation = nn.GELU()
#         self.out_layer2 = Conv1D(in_dim=dim,
#                                  out_dim=dim,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0,
#                                  bias=True)
        
#         self.p1_out_layer = Conv1D(in_dim=dim,
#                                  out_dim=dim,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0,
#                                  bias=True)
        
#         self.p2_out_layer = Conv1D(in_dim=dim,
#                                  out_dim=dim,
#                                  kernel_size=1,
#                                  stride=1,
#                                  padding=0,
#                                  bias=True)


#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1,
#                          3)  # (batch_size, num_heads, w_seq_len, head_size)

#     @staticmethod
#     def combine_last_two_dim(x):
#         old_shape = list(x.size())
#         new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
#         return x.reshape(shape=new_shape)

#     def forward(self, x, p1, p2=None, prior_1=None, prior_2=None, mask=None):
        
#         p1_len = p1.shape[1]
#         p2_len = p2.shape[1]
#         prompt_len = p1_len + p2_len
#         T = x.shape[1]

#         # output = torch.cat([x, p1, p2], dim=1)
#         output = self.layer_norm1(x)  # (batch_size, seq_len, dim)

#         # multi-head attention layer
#         query = self.transpose_for_scores(
#             torch.cat([self.query(x), p1, p2], dim=1))  # (batch_size, num_heads, seq_len, head_size)
#         key = self.transpose_for_scores(torch.cat([self.key(x), p1, p2], dim=1))
        
#         value = self.transpose_for_scores(self.value(x))


#         attention_scores = torch.matmul(query, key.transpose(
#             -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
#         attention_scores = attention_scores / math.sqrt(self.head_size)

#         # p1_attn_scores = attention_scores[:, :, -prompt_len:-p2_len, :-prompt_len]
#         # p2_attn_scores = attention_scores[:, :, -p2_len:, :-prompt_len]
#         # word_attn_scores = attention_scores[:, :, :-prompt_len, :-prompt_len]

#         p1_attn_scores = attention_scores[:, :,  :-prompt_len, -prompt_len:-p2_len].transpose(-1, -2)
#         p2_attn_scores = attention_scores[:, :, :-prompt_len, -p2_len:].transpose(-1, -2)
#         word_attn_scores = attention_scores[:, :, :-prompt_len, :-prompt_len]
        

#         if mask is not None:  # masking
#             mask = mask.unsqueeze(1).unsqueeze(
#                 2)  # (batch_size, 1, 1, seq_len)
#             word_attn_scores = mask_logits(word_attn_scores, mask)

#         # softmax
#         # p1_attn_scores [batch, num_heads, num_prompt, T]
#         p1_attn_scores = F.softmax(F.softmax(p1_attn_scores, dim=-1)*prior_1.unsqueeze(1).unsqueeze(2), dim=-1)  # (batch_size, num_heads, seq_len, seq_len))
#         p2_attn_scores = F.softmax(F.softmax(p2_attn_scores, dim=-1)*prior_2.unsqueeze(1).unsqueeze(2), dim=-1)
#         # p1_attn_scores = F.softmax(p1_attn_scores, dim=-1)
#         # p2_attn_scores = F.softmax(p2_attn_scores, dim=-1)
#         word_attn_scores = F.softmax(word_attn_scores, dim=-1)

#         p1_attn_scores = self.dropout(p1_attn_scores)
#         p2_attn_scores = self.dropout(p2_attn_scores)
#         word_attn_scores = self.dropout(word_attn_scores)

#         value = torch.matmul(word_attn_scores, value)  # (batch_size, num_heads, seq_len, head_size)
#         p1_v = torch.matmul(p1_attn_scores, value)
#         p2_v = torch.matmul(p2_attn_scores, value)


#         value = self.combine_last_two_dim(value.permute(
#             0, 2, 1, 3))  # (batch_size, seq_len, dim)
#         p1_v = self.combine_last_two_dim(p1_v.permute(
#             0, 2, 1, 3))  # (batch_size, seq_len, dim)
#         p2_v = self.combine_last_two_dim(p2_v.permute(
#             0, 2, 1, 3))  # (batch_size, seq_len, dim)

#         # intermediate layer
#         output = self.dropout(value)
#         residual = x + output
#         output = self.layer_norm2(residual)
#         output = self.out_layer1(output)
#         output = self.output_activation(output)
#         output = self.dropout(output)
#         output = self.out_layer2(output) + residual

#         p1_v = self.dropout(p1_v)
#         p1_out = p1_v + p1
#         p1_out = self.layer_norm2(p1_out)
#         p1_out = self.p1_out_layer(p1_out)
#         p1_out = self.output_activation(p1_out)
#         p1_out = self.dropout(p1_out)

#         p2_v = self.dropout(p2_v)
#         p2_out = p2_v + p2
#         p2_out = self.layer_norm2(p2_out)
#         p2_out = self.p2_out_layer(p2_out)
#         p2_out = self.output_activation(p2_out)
#         p2_out = self.dropout(p2_out)

#         return output, p1_out, p2_out
    


class PriorGuidedPromptAttention(nn.Module):
    def __init__(self, configs):
        super(PriorGuidedPromptAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        self.key = Conv1D(in_dim=dim,
                          out_dim=dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.value = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)

        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        
        self.p1_out_layer = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        
        self.p2_out_layer = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, p1, p2=None, prior_1=None, prior_2=None, mask=None):
        
        p1_len = p1.shape[1]
        
        if p2 is not None:
            p2_len = p2.shape[1]
            prompt_len = p1_len + p2_len
        else:
            prompt_len = p1_len
        
        T = x.shape[1]

        # output = torch.cat([x, p1, p2], dim=1)
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)

        # multi-head attention layer
        
        if p2 is not None:
            query = self.transpose_for_scores(
                torch.cat([self.query(x), p1, p2], dim=1))  # (batch_size, num_heads, seq_len, head_size)
            key = self.transpose_for_scores(torch.cat([self.key(x), p1, p2], dim=1))
        else:
            query = self.transpose_for_scores(
                torch.cat([self.query(x), p1], dim=1))  # (batch_size, num_heads, seq_len, head_size)
            key = self.transpose_for_scores(torch.cat([self.key(x), p1], dim=1))
        
        value = self.transpose_for_scores(self.value(x))


        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)

        # p1_attn_scores = attention_scores[:, :, -prompt_len:-p2_len, :-prompt_len]
        # p2_attn_scores = attention_scores[:, :, -p2_len:, :-prompt_len]
        # word_attn_scores = attention_scores[:, :, :-prompt_len, :-prompt_len]
        if p2 is not None:
            p1_attn_scores = attention_scores[:, :,  :-prompt_len, -prompt_len:-p2_len].transpose(-1, -2)
            p2_attn_scores = attention_scores[:, :, :-prompt_len, -p2_len:].transpose(-1, -2)
        else:
            p1_attn_scores = attention_scores[:, :, :-prompt_len, -prompt_len:].transpose(-1, -2)
            p2_attn_scores = None

        word_attn_scores = attention_scores[:, :, :-prompt_len, :-prompt_len]
        

        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            word_attn_scores = mask_logits(word_attn_scores, mask)

        # softmax
        # p1_attn_scores [batch, num_heads, num_prompt, T]
        p1_attn_scores = F.softmax(F.softmax(p1_attn_scores, dim=-1)*prior_1.unsqueeze(1).unsqueeze(2), dim=-1)  # (batch_size, num_heads, seq_len, seq_len))
        p1_attn_scores = self.dropout(p1_attn_scores)
        
        if p2_attn_scores is not None:
            p2_attn_scores = F.softmax(F.softmax(p2_attn_scores, dim=-1)*prior_2.unsqueeze(1).unsqueeze(2), dim=-1)
            p2_attn_scores = self.dropout(p2_attn_scores)
        
        word_attn_scores = F.softmax(word_attn_scores, dim=-1)
        word_attn_scores = self.dropout(word_attn_scores)

        value = torch.matmul(word_attn_scores, value)  # (batch_size, num_heads, seq_len, head_size)
        p1_v = torch.matmul(p1_attn_scores, value)
       
        if p2 is not None:
            p2_v = torch.matmul(p2_attn_scores, value)
        else:
            p2_v = None

        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        p1_v = self.combine_last_two_dim(p1_v.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)

        if p2_v is not None:
            p2_v = self.combine_last_two_dim(p2_v.permute(
                0, 2, 1, 3))  # (batch_size, seq_len, dim)

        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual

        p1_v = self.dropout(p1_v)
        p1_out = p1_v + p1
        p1_out = self.layer_norm2(p1_out)
        p1_out = self.p1_out_layer(p1_out)
        p1_out = self.output_activation(p1_out)
        p1_out = self.dropout(p1_out)

        if p2 is not None:
            p2_v = self.dropout(p2_v)
            p2_out = p2_v + p2
            p2_out = self.layer_norm2(p2_out)
            p2_out = self.p2_out_layer(p2_out)
            p2_out = self.output_activation(p2_out)
            p2_out = self.dropout(p2_out)
        else:
            p2_out = None

        return output, p1_out, p2_out
    

class MultiLSTMAttention(nn.Module):
    def __init__(self, configs):
        super(MultiLSTMAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        num_layers = configs.num_layers
        num_step = configs.num_step
        bi_direction = configs.bi_direction

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = MultiStepLSTMEncoder(in_dim=dim,
                                          out_dim=dim,
                                          num_layers=num_layers,
                                          num_step=num_step,
                                          bi_direction=bi_direction,
                                          drop_rate=drop_rate)
        self.key = MultiStepLSTMEncoder(in_dim=dim,
                                        out_dim=dim,
                                        num_layers=num_layers,
                                        num_step=num_step,
                                        bi_direction=bi_direction,
                                        drop_rate=drop_rate)
        self.value = MultiStepLSTMEncoder(in_dim=dim,
                                          out_dim=dim,
                                          num_layers=num_layers,
                                          num_step=num_step,
                                          bi_direction=bi_direction,
                                          drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output


class MultiConvAttention(nn.Module):
    def __init__(self, configs):
        super(MultiConvAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        kernels = configs.kernels

        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = TemporalContextModule(in_dim=dim,
                                           out_dim=dim,
                                           kernels=kernels,
                                           drop_rate=drop_rate)
        self.key = TemporalContextModule(in_dim=dim,
                                         out_dim=dim,
                                         kernels=kernels,
                                         drop_rate=drop_rate)
        self.value = TemporalContextModule(in_dim=dim,
                                           out_dim=dim,
                                           kernels=kernels,
                                           drop_rate=drop_rate)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output