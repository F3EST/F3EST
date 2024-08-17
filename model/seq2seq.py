import math
import random
import copy
import time
import operator
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import Optional
from .blocks import (TransformerBlock, MaskedConv1D, ConvBlock, LayerNorm, LocalMHA, FeedForward)
from queue import PriorityQueue


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.2,
                 maxlen: int = 200):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        # pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        # pos_embedding = torch.zeros((maxlen, emb_size))
        self.pos_embedding = nn.Parameter(torch.zeros(1, maxlen, emb_size), requires_grad=False)
        self.pos_embedding[:, :, 0::2] = torch.sin(pos * den)
        self.pos_embedding[:, :, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)].detach())


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Encoder Transformer
class EncoderTransformer(nn.Module):
    def __init__(self, feat_dim, d_model, nhead=8, dim_feedforward=512, num_layers=3, dropout=0.1, activation='gelu',
                 kernal_size=3, use_conv=True, device=None):
        super(EncoderTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.gelu = nn.GELU()
        if use_conv:
            self.conv = nn.Conv1d(d_model, d_model, kernal_size, stride=1, padding=kernal_size // 2)

        else:
            self.conv = None

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        if self.conv is not None:
            src = self.gelu(self.conv(src.transpose(1, 2)).transpose(1, 2))
        return src
        src = self.positional_encoding(src)
        return self.encoder(src, mask, src_key_padding_mask, is_causal)


# Sequence Encoder
class Encoder(nn.Module):
    def __init__(self, num_classes, d_model, nhead=8, dim_feedforward=512, num_layers=3, dropout=0.1, activation='gelu',
                 device=None):
        super(Encoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.gelu = nn.GELU()
        self.tok_emb = nn.Linear(num_classes + 2, d_model)
        self.generator = nn.Linear(d_model, num_classes)
        self.cls = nn.Linear(d_model, 2)

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        src = self.tok_emb(src)
        src = self.positional_encoding(src)
        src = self.encoder(src, mask, src_key_padding_mask, is_causal)
        return self.generator(self.gelu(src)), self.cls(src)


# Decoder Transformer
class DecoderTransformer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size, nhead=8, dim_feedforward=512, num_layers=3, dropout=0.1,
                 activation='gelu', device=None):
        super(DecoderTransformer, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   activation=activation,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt = self.tgt_tok_emb(tgt)
        tgt = self.positional_encoding(tgt)
        outs_feat = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                 memory_key_padding_mask=memory_key_padding_mask)
        outs = self.generator(outs_feat)
        return outs, outs_feat

