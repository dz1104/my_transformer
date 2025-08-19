import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import os
import math
import copy





class PositionalEncoding(nn.Module):
    """
    Implements the positional encoding as described in the "Attention is All You Need" paper.
    hidden_size: 模型的维度，必须是偶数
    max_len: 预先计算的最大序列长度
    dropout: Dropout层的比率
    """

    # 位置编码的形状为 (max_len, hidden_size)
    def __init__(self, hidden_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)

        # 创建一个底板tensor，作为填入位置编码的容器
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # shape = (max_len,1)

        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000) / hidden_size))
        # shape = (hidden_size/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 在第0维增加一个维度，变成 (1, max_len, hidden_size)

        self.register_buffer('pe', pe)  # 将pe注册为buffer，这样它不会被视为模型参数，但会被保存和加载

    def forward(self, x):
        """
        x: 输入的tensor，形状为 (batch_size, seq_len, hidden_size)
        返回位置编码后的tensor，形状为 (batch_size, seq_len, hidden_size)
        """
        x = x + self.pe[:, x.size(1), :]
        return x  # 应用dropout层
class Embedding_layer(nn.Module):
    def __init__(self,vocab_size,hidden_size,dropout=0.1,max_len=5000,layernorm_eps=1e-12):
        super(Embedding_layer, self).__init__()
        self.embedding = nn.Embedding(vocab_size,hidden_size)
        self.pe = PositionalEncoding(max_len=max_len, hidden_size=hidden_size)
        self.dropout=nn.Dropout(p=dropout)
        self.layernorm =nn.LayerNorm(hidden_size,eps=layernorm_eps)

    def forward(self, input_ids):
        input_emb = self.embedding(input_ids)
        input_emb = self.pe(input_emb)
        input_emb = self.dropout(input_emb)
        # input_emb = self.layernorm(input_emb)
        return input_emb
def scaled_dot_product_attention(q, k, v, mask=None):
    """
        计算缩放点积注意力
        Args:
            q (torch.Tensor): 查询张量, 形状 (..., seq_len_q, d_k)
            k (torch.Tensor): 键张量, 形状 (..., seq_len_k, d_k)
            v (torch.Tensor): 值张量, 形状 (..., seq_len_v, d_v), seq_len_k == seq_len_v
            mask (torch.Tensor, optional): 掩码张量, 形状 (..., seq_len_q, seq_len_k). Defaults to None.
        Returns:
            torch.Tensor: 输出张量
            torch.Tensor: 注意力权重
    """
    d_k = q.size(-1)
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(score, dim=-1)
    output = torch.matmul(p_attn, v)

    return output, p_attn



class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)



    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch,num_heads,seq_len,head_dim]

        output, p_attn = scaled_dot_product_attention(q, k, v, mask)

        output = output.view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.w_o(output)
        output = self.dropout(output)

        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self,hidden_size,ffn_dim,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(hidden_size,ffn_dim)
        self.w_2 = nn.Linear(ffn_dim,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.w_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x
class SublayerConnect(nn.Module):
    def __init__(self,hidden_size,dropout=0.1):
        super(SublayerConnect,self).__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        residual_output = sublayer(x) + x
        return self.norm(residual_output)

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads,ffn):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.ffn = PositionwiseFeedForward(hidden_size=hidden_size, ffn_dim=ffn)
        self.sublayer = nn.ModuleList(SublayerConnect(hidden_size=hidden_size) for _ in range(2))
    def forward(self, x, mask):
        x = self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        x = self.sublayer[1](x , self.ffn)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim):
        super(DecoderLayer, self).__init__()
        self.mask_attn = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.ffn = PositionwiseFeedForward(hidden_size=hidden_size, ffn_dim=ffn_dim)
        self.cross_attn = MultiHeadAttention(hidden_size=hidden_size, num_heads=num_heads)
        self.sublayer = nn.ModuleList(SublayerConnect(hidden_size=hidden_size) for _ in range(3))

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.mask_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))
        x = self.sublayer[2](x, self.ffn)
        return x

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, ffn):
        super(Encoder, self).__init__()
        self.layer = EncoderLayer(hidden_size=hidden_size,num_heads=num_heads,ffn=ffn)
        self.Encoder = clones(self.layer, num_layers)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask):
        for layer in self.Encoder:
            x = layer(x, mask)
        return self.LayerNorm(x)

class Decoder(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, ffn):
        super(Decoder, self).__init__()
        self.layer = DecoderLayer(hidden_size=hidden_size, num_heads=num_heads, ffn_dim=ffn)
        self.Decoder = clones(self.layer, num_layers)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.Decoder:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.LayerNorm(x)

class Predict(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(Predict, self).__init__()
        self.predict_layer = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.predict_layer(x)
        return F.log_softmax(x, dim=-1)




