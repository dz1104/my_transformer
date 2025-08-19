import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import os
import math
from Module import Embedding_layer,Encoder, Decoder,Predict

import copy

vocab_size = 51200
en_seq_len = 144
de_seq_len = 72
hidden_size = 512
batch_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, hidden_size, num_layers,num_heads,ffn,
                 dropout=0.1):
        super(transformer, self).__init__()
        self.en_embedding = Embedding_layer(vocab_size=src_vocab, hidden_size=hidden_size, dropout=dropout)
        self.de_embedding = Embedding_layer(vocab_size=tgt_vocab, hidden_size=hidden_size, dropout=dropout)
        self.encoder = Encoder(hidden_size=hidden_size, num_layers=num_layers,num_heads=num_heads, ffn=ffn)
        self.decoder = Decoder(hidden_size=hidden_size, num_layers=num_layers,num_heads=num_heads, ffn=ffn)
        self.predict = Predict(hidden_size=hidden_size, vocab_size=tgt_vocab)

    def make_src_mask(self, src, pad_idx=0):

        src_mask = (src != pad_idx)

        return src_mask.unsqueeze(1).unsqueeze(2)
    def make_tgt_mask(self, tgt, pad_idx=0):
        batch_size, tgt_len = tgt.shape[:2]
        device = tgt.device

        # 1. 创建padding掩码
        # 形状: (batch_size, 1, 1, tgt_len)
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

        # 2. 创建look-ahead掩码
        # 形状: (tgt_len, tgt_len)
        lookahead_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=device)).bool()

        # 3. 合并两个掩码
        # tgt_pad_mask广播为 (batch_size, 1, 1, tgt_len)
        # lookahead_mask广播为 (1, 1, tgt_len, tgt_len) -> (batch_size, 1, tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & lookahead_mask

        return tgt_mask



    def forward(self, src_tokens, tgt_tokens):
        src_mask = self.make_src_mask(src_tokens)
        tgt_mask = self.make_tgt_mask(tgt_tokens)
        memory = self.en_embedding(src_tokens)
        x = self.de_embedding(tgt_tokens)
        memory = self.encoder(memory, src_mask)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.predict(x)

    def encode(self, src_tokens):
        src_mask = self.make_src_mask(src_tokens)
        return self.encoder(self.en_embedding(src_tokens), src_mask)

    def decode(self, src_tokens, tgt_tokens):
        src_mask = self.make_src_mask(src_tokens)
        tgt_mask = self.make_tgt_mask(tgt_tokens)
        return self.decoder(self.de_embedding(tgt_tokens), self.en_embedding(src_tokens), src_mask, tgt_mask)

    def predict_next_word(self, src_tokens, tgt_tokens):
        out = self.decode(src_tokens, tgt_tokens)
        prob = self.predict(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        return next_word.item()

def make_model(src_vocab, tgt_vocab, hidden_size, num_layers, num_heads, ffn, device, dropout=0.1):
    model = transformer(src_vocab, tgt_vocab, hidden_size, num_layers, num_heads=num_heads, ffn=ffn,dropout=dropout)
    model.to(device)
    return model




if __name__ == '__main__':

    # Encoder_tokens = torch.randint(0, vocab_size, (batch_size, en_seq_len)).to(device)
    # Decoder_tokens = torch.randint(0, vocab_size, (batch_size, de_seq_len)).to(device)
    transformer = make_model(src_vocab=11,tgt_vocab=11,hidden_size=hidden_size, num_layers=6, num_heads=8, ffn=hidden_size*4,device=device, dropout=0.1)

    x = torch.randint(1, 11, size=(1, 20), device=device)
    y = torch.ones(1, 1).to(device).type_as(x)

    next_word = transformer.predict_next_word(x, y)

    # x = transformer(Encoder_tokens, Decoder_tokens)

    print(next_word)



