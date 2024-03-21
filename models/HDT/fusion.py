import torch
import torch.nn as nn
import math
import numpy as np


from .operation import Conv1D, mask_logits


class CQFusion(nn.Module):
    def __init__(self, dim, mode = 'entity', drop_rate=0.0):
        super(CQFusion, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def forward(self, context, query, bridge, c_mask, q_mask):
        score = self.trilinear_attention(
            context, query)  # (batch_size, c_seq_len, q_seq_len)
        score_ = torch.softmax(mask_logits(score, q_mask.unsqueeze(1)),
                               dim=2)  # (batch_size, c_seq_len, q_seq_len)
        score_t = torch.softmax(mask_logits(score, c_mask.unsqueeze(2)),
                                dim=1)  # (batch_size, c_seq_len, q_seq_len)
        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)
        q2c = torch.matmul(torch.matmul(score_, score_t),
                           context)  # (batch_size, c_seq_len, dim)
        output = torch.cat(
            [context, c2q,
             torch.mul(context, c2q),
             torch.mul(context, q2c)],
            dim=2)
        output = self.cqa_linear(output)  # (batch_size, c_seq_len, dim)
        return output * c_mask.unsqueeze(2)

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand(
            [-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand(
            [-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))
        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)
        return res



class BridgetFusion(nn.Module):
    def __init__(self, dim, mode = 'entity', drop_rate=0.0, ):
        super(BridgetFusion, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.cqa_linear = Conv1D(in_dim=4 * dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        
        num_heads = 8

        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        
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

        self.layer_norm = nn.LayerNorm(dim)
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
        
        self.mode = mode

        self.index = 0

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


    def forward(self, context, query, bridge, c_mask, q_mask):
        # word fusion
        score = self.trilinear_attention(
            context, query)  # (batch_size, c_seq_len, q_seq_len)

        # print(score.shape)
        score_ = torch.softmax(mask_logits(score, q_mask.unsqueeze(1)),
                               dim=2)  # (batch_size, c_seq_len, q_seq_len)

        score_t = torch.softmax(mask_logits(score, c_mask.unsqueeze(2)),
                                dim=1)  # (batch_size, c_seq_len, q_seq_len)
        
        # ---------------------------- just for test ------------------------------------------- 
        # if self.mode == 'motion':
        #     np.save('test/only_motion/' + str(self.index), score_.detach().cpu().numpy())
        # self.index += 1
        # ---------------------------- just for test -------------------------------------------

        score_t = score_t.transpose(1, 2)  # (batch_size, q_seq_len, c_seq_len)
        c2q = torch.matmul(score_, query)  # (batch_size, c_seq_len, dim)

        

        q2c = torch.matmul(torch.matmul(score_, score_t),
                           context)  # (batch_size, c_seq_len, dim)
        
        word_fusion = torch.cat(
            [context, c2q,
             torch.mul(context, c2q),
             torch.mul(context, q2c)],
            dim=2)
        word_fusion = self.cqa_linear(word_fusion) * c_mask.unsqueeze(2)  # (batch_size, c_seq_len, dim)

        # semantic fusion
        word_fusion_norm = self.layer_norm(word_fusion)
        bridge_norm = self.layer_norm(bridge)

        
        q = self.transpose_for_scores(self.query(word_fusion_norm))
        k = self.transpose_for_scores(self.key(bridge_norm))
        v = self.transpose_for_scores(self.value(bridge_norm))

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        # attention_scores = mask_logits(attention_scores, c_mask)
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # print(attention_probs.shape)    
        v = torch.matmul(attention_probs, v) 

        v = self.combine_last_two_dim(v.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        
        output = self.dropout(v)
        residual = word_fusion + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual

        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand(
            [-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand(
            [-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))

        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)  + subres1 
        return res
    

class SemanticFusion(nn.Module):
    def __init__(self, dim, mode = 'entity', drop_rate=0.0, ):
        super(SemanticFusion, self).__init__()
        w4C = torch.empty(dim, 1)
        w4Q = torch.empty(dim, 1)
        w4mlu = torch.empty(1, 1, dim)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C, requires_grad=True)
        self.w4Q = nn.Parameter(w4Q, requires_grad=True)
        self.w4mlu = nn.Parameter(w4mlu, requires_grad=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        
        num_heads = 8

        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        
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

        self.layer_norm = nn.LayerNorm(dim)
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
        
        self.mode = mode

        self.index = 0

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


    def forward(self, context, query, bridge, c_mask, q_mask):
        semantic_fusion = self.linear(context)

        # semantic fusion
        semantic_fusion_norm = self.layer_norm(semantic_fusion)
        bridge_norm = self.layer_norm(bridge)

        
        q = self.transpose_for_scores(self.query(semantic_fusion_norm))
        k = self.transpose_for_scores(self.key(bridge_norm))
        v = self.transpose_for_scores(self.value(bridge_norm))

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        # attention_scores = mask_logits(attention_scores, c_mask)
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # print(attention_probs.shape)    
        v = torch.matmul(attention_probs, v) 

        v = self.combine_last_two_dim(v.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        
        output = self.dropout(v)
        residual = semantic_fusion_norm + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual

        return output

    def trilinear_attention(self, context, query):
        batch_size, c_seq_len, dim = context.shape
        batch_size, q_seq_len, dim = query.shape
        context = self.dropout(context)
        query = self.dropout(query)
        subres0 = torch.matmul(context, self.w4C).expand(
            [-1, -1, q_seq_len])  # (batch_size, c_seq_len, q_seq_len)
        subres1 = torch.matmul(query, self.w4Q).transpose(1, 2).expand(
            [-1, c_seq_len, -1])
        subres2 = torch.matmul(context * self.w4mlu, query.transpose(1, 2))

        res = subres0 + subres1 + subres2  # (batch_size, c_seq_len, q_seq_len)  + subres1 
        return res