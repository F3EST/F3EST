import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from .common import SingleStageTCN
from .impl.asformer import MyTransformer
from .impl.gtad import GCNeXt
from .impl.actionformer import ConvTransformerBackbone, FPN1D, PtTransformerClsHead

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class FCPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        batch_size, clip_len, _ = x.shape
        return self._fc_out(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class VideoClassifier(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        self._fc = nn.Linear(feat_dim, feat_dim)
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self._dropout = nn.Dropout()
        self._relu = nn.ReLU()

    def forward(self, x):
        x = self._dropout(self._relu(self._fc(x)))
        return self._fc_out(x)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, num_classes)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class GRU(nn.Module):

    def __init__(self, feat_dim, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc_out = FCPrediction(2 * hidden_dim, hidden_dim)
        self._dropout = nn.Dropout()

    def forward(self, x):
        y, _ = self._gru(x)
        return self._fc_out(self._dropout(y))


class GRU_FC(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim, num_layers=1):
        super().__init__()
        self._gru = nn.GRU(
            feat_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True)
        self._fc1 = nn.Linear(in_features=2 * hidden_dim + 9, out_features=num_classes)
        self._fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        self._dropout = nn.Dropout()
        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()

    def forward(self, x, k):
        y, _ = self._gru(x)
        y = self._dropout(y[:, -1, :])
        cat = torch.cat([k, y], 1)
        return self._fc1(cat)


class TCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_stages=1, num_layers=5):
        super().__init__()

        self._tcn = SingleStageTCN(
            feat_dim, 256, num_classes, num_layers, True)
        self._stages = None
        if num_stages > 1:
            self._stages = nn.ModuleList([SingleStageTCN(
                num_classes, 256, num_classes, num_layers, True)
                for _ in range(num_stages - 1)])

    def forward(self, x):
        x = self._tcn(x)
        if self._stages is None:
            return x
        else:
            outputs = [x]
            for stage in self._stages:
                x = stage(F.softmax(x, dim=2))
                outputs.append(x)
            return torch.stack(outputs, dim=0)


class ASFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, num_decoders=3, num_layers=5):
        super().__init__()

        r1, r2 = 2, 2
        num_f_maps = 64
        self._net = MyTransformer(
            num_decoders, num_layers, r1, r2, num_f_maps, feat_dim,
            num_classes, channel_masking_rate=0.3)

    def forward(self, x):
        B, T, D = x.shape
        return self._net(
            x.permute(0, 2, 1), torch.ones((B, 1, T), device=x.device)
        ).permute(0, 1, 3, 2)
    
    
class GCNPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, hidden_dim=256, num_layers=2):
        super().__init__()

        self.idx_list = []
        self.fc_in = nn.Linear(feat_dim, hidden_dim)
        gcn_layers = [GCNeXt(hidden_dim, hidden_dim, k=3, groups=32,
                             idx=self.idx_list) for _ in range(num_layers)]
        self.backbone = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=4),
            nn.ReLU(inplace=True),
            *gcn_layers
        )
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        del self.idx_list[:]
        batch_size, clip_len, _ = x.shape
        x = self.fc_in(x.view(batch_size * clip_len, -1))
        x = F.relu(x).view(batch_size, clip_len, -1)
        x = self.backbone(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x)
        return self.fc(x.reshape(batch_size * clip_len, -1)).view(
            batch_size, clip_len, -1)


class ActionFormerPrediction(nn.Module):

    def __init__(self, feat_dim, num_classes, kernal_size=3, d_model=256, n_head=4, max_len=128):
        super().__init__()

        self.backbone = ConvTransformerBackbone(feat_dim, d_model, n_head, 3, max_len)
        self.neck = FPN1D([d_model], d_model)
        self.cls_head = PtTransformerClsHead(d_model, d_model, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, C, T = x.size()
        batch_masks = torch.ones((B, 1, T), device=x.device, dtype=torch.bool)
        feats, masks = self.backbone(x, batch_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)
        return out_cls_logits[0].transpose(1, 2)
        

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
