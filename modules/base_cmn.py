from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .att_model import pack_wrapper, AttModel


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

#attention有两种，addtitive attention和dot-product attention
 # 类似于，这里假定query来自target language sequence；
 #        # key和value都来自source language sequence.
def attention(query, key, value, mask=None, dropout=None):
    # query：[8, 8，不定49/98, 64]  key：[8, 8, 98, 64] value:[8, 8, 98, 64]
    #query/key,value:(batch.size,head.num,目标序列中词的个数/当前序列的词的个数,64是每个词对应的向量表示)
    d_k = query.size(-1)   #64
    # 代表98个目标语言序列中每个词和98个源语言序列的分别的“亲密度”。
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) #[8, 8, 98, 64]*[8, 8, 64, 98]注意是最后两个维度相乘,得到(8,8,98,98)，
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)  #对scores的最后一个维度执行softmax，得到的还是一个tensor, (8,8,98,98)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意，这里返回p_attn主要是用来可视化显示多头注意力机制。
    #(8,8,98,98)乘以[8, 8, 98, 64]->[8, 8, 98, 64]
    return torch.matmul(p_attn, value), p_attn

#-------------------------CMN加入
def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=8):
    # query：[8, 8，不定49/98, 64]  key：[8, 8, 2048, 64] value:[8, 8, 2048, 64]
    #d_k: 64
    d_k = query.size(-1)
    #scores:[8, 8, 不定, 2048]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  #Dsi = qs · k / √ d   (7)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    #selected_scores, idx:[8, 8, 不定, 32]
    selected_scores, idx = scores.topk(topk)
    # dummy_value:([8, 8, 不定, 2048, 64])
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    #dummy_idx:([8, 8, 不定, 32, 64])
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    #selected_value:([8, 8, 不定, 32, 64])----value里选出idx的
    selected_value = torch.gather(dummy_value, 3, dummy_idx)

    #p_attn:([8, 8, 不定, 32])
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    #out1:torch.Size([8, 8, 不定, 64])，([8, 8, 不定, 32])
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module): #EncoderDecoder
    def __init__(self, encoder, decoder, src_embed, tgt_embed, cmn):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 源语言序列的编码，包括词嵌入和位置编码(视觉响应）
        self.src_embed = src_embed
        # 目标语言序列的编码，包括词嵌入和位置编码（生成的报告）
        self.tgt_embed = tgt_embed
        self.cmn = cmn

        self.fuse_feature = nn.Linear(512 * 2, 512)

    # 输入：视觉响应att_feats：([8, 98, 512]) seq：torch.Size([8, 不定-1]),att_masks：([8, 1, 98]),seq_mask:([8, X, X])，memory_matrix
    def forward(self, src, tgt, src_mask, tgt_mask, memory_matrix):
        #decoder输入：视觉响应的resulted intermediate states + 前一步生成文本的文本响应
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask, memory_matrix=memory_matrix)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, past=None, memory_matrix=None):
        embeddings = self.tgt_embed(tgt)   #(16,59/51,512)

        # Memory querying and responding for textual features
        # dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0), memory_matrix.size(1))
        dummy_memory_matrix = memory_matrix
        if len(memory_matrix.shape) == 2:
            dummy_memory_matrix = memory_matrix.unsqueeze(0).expand(embeddings.size(0), memory_matrix.size(0), memory_matrix.size(1))
        #从上一步骤生成报告的文本响应  responses：torch.Size([16, 59, 512])
        responses = self.cmn(embeddings, dummy_memory_matrix, dummy_memory_matrix)
        embeddings = embeddings + responses       #特征交互：简单相加
        # 特征交互，不是简单相加，先concat再经过Linear层
        #embeddings：torch.Size([8, 59, 512])
        # embeddings = self.fuse_feature(torch.cat((embeddings, responses), dim=2))  # {FCN}(Concat）（14）
        # Memory querying and responding for textual features
        return self.decoder(embeddings, memory, src_mask, tgt_mask, past=past)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        #包含3个-----Transformer EncoderLayer forward------每个里有1次MultiHeadedAttention
        for layer in self.layers:
            x = layer(x, mask)
        # 最后做一次LayerNorm，最后的输出也是(8, 98, 512) shape
        return self.norm(x)

#SubLayerConnection 子层连接
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        # features=d_model=512, eps=epsilon 用于分母的非0化平滑
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))    #gamma可训练参数向量
        self.b_2 = nn.Parameter(torch.zeros(features))   #beta可训练参数向量
        self.eps = eps

    def forward(self, x):
        # x 的形状为(batch.size, sequence.len, 512)
        # 就是在统计每个样本所有维度的值，求均值和方差，所以就是在hidden dim上操作
        # 对x的最后一个维度，取平均值，得到tensor (batch.size, seq.len)
        mean = x.mean(-1, keepdim=True)
        # 对x的最后一个维度，取标准方差，得(batch.size, seq.len)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
       A residual connection followed by a layer norm.
       Note for code simplicity the norm is first as opposed to last.
       """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        # size=d_model=512，self_attn = MultiHeadAttention对象
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 使用深度克隆方法，完整地复制出来两个SublayerConnection
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # x shape = (8, 98, 512)  x的shape不变
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
#----------这里有不一样
    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        #包含3个----Transformer DecoderLayer forward------------------每个里有2次MultiHeadedAttention
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        # self_attn = one MultiHeadAttention object，目标语言序列的
        # src_attn = second MultiHeadAttention object, 目标语言序列
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        #需要三个SublayerConnection, 分别在self.self_attn, self.src_attn, 和self.feed_forward的后边

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory # (batch.size, sequence.len, 512)
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask)) #实现目标序列的自注意力编码
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask)) #实现目标序列和源序列的注意力计算
            return self.sublayer[2](x, self.feed_forward) #走一个全连接层，然后
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present

#在CMN的操作里
class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):   #3,512
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h    #170
        self.h = h              #3
        self.linears = clones(nn.Linear(d_model, d_model), 4)  #in_features: 512, out_features: 512
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)      #16，batch_size
        #query：[8, 98, 512]  key：[8, 2048, 512] value:[8, 2048, 512]----att_feats, dummy_memory_matrix, dummy_memory_matrix

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:  #is none
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
            #query：[8, 不定, 512]  key：[8, 2048, 512] value:[8, 2048, 512]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        # query：[8, 8，不定, 64]  key：[8, 8, 2048, 64] value:[8, 8, 2048, 64] 维度都变成64
        # 类似于，这里假定query来自target language sequence；
        # key和value都来自source language sequence
        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        #----------memory query、responding--------------------------------------
        # x：[8, 8，不定, 64]  self.attn：[8, 8, 不定, 32]---torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn
        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)
        # [8, 不定, 512]
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            #[8, 不定, 512]
            return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h   #512 // 8 = 64
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) #定义四个Linear networks, 每个的大小是(512, 512)的，
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        #输入的query：[8, 98（target.seq.len）, 512]  key：[8, 98（src.seq.len）, 512] value:[8, 98（src.seq.len）, 512]
        # 类似于，这里假定query来自target language sequence；
        # key和value都来自source language sequence.
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            # 这里是前三个Linear Networks的具体应用，
            # 例如query=[8, 98, 512] -> Linear network -> [8, 98, 512]
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        # -> view -> (8,98, 8, 64) -> transpose(1,2) -> (8, 8, 98, 64)
        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        #输出的x形状为（8, 8, 98, 64）
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        #x ~ （8, 8, 98, 64） -> transpose(1,2) -> (8, 98, 8, 64) -> contiguous() and view -> (8, 98, 512)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            # 执行第四个Linear network，把(8, 98, 512)经过一次linear network，得到(8, 98, 512)即(batch.size, sequence.length, d_model)
            return self.linears[-1](x)

#全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model = 512  d_ff = 512
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape = (batch.size, sequence.len, 512)例如(8, 98, 512)
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        # d_model=512, vocab=当前语言的词表大小=761
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # x ~ (batch.size, sequence.length, one-hot),-x :torch.Size([8, 59])
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
        position = torch.arange(0, max_len).unsqueeze(1).float()      # (5000) -> (5000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)# 偶数下标的位置
        pe[:, 1::2] = torch.cos(position * div_term)# 奇数下标的位置
        pe = pe.unsqueeze(0)# (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)] #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
        return self.dropout(x)


class BaseCMN(AttModel):

    #模块的init
    def make_model(self, tgt_vocab, cmn):
        c = copy.deepcopy
        #MultiHeadedAttention init
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        #Transformer init
        model = Transformer(
            # memory responses of visual and textual features are functionalized as the input of the encoder and decoder
            Encoder(EncoderLayer(self.d_model, c(attn), c(ff), self.dropout), self.num_layers),
            Decoder(DecoderLayer(self.d_model, c(attn), c(attn), c(ff), self.dropout), self.num_layers),
            nn.Sequential(c(position)),
            nn.Sequential(Embeddings(self.d_model, tgt_vocab), c(position)), cmn)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, tokenizer):
        super(BaseCMN, self).__init__(args, tokenizer)
        self.args = args
        self.num_layers = args.num_layers   #the number of layers of Transformer:3
        self.d_model = args.d_model        #512
        self.d_ff = args.d_ff       #512
        self.num_heads = args.num_heads  #the number of heads in Transformer:8
        self.dropout = args.dropout    #0.1
        self.topk = args.topk       #the number of k:32--->16

        #加入特征融合，不是简单相加
        self.fuse_feature = nn.Linear(args.d_model * 2, args.d_model)

        #tgt_vocab:761
        tgt_vocab = self.vocab_size + 1

        #self.cmn:  4个: Linear(in_features=512, out_features=512, bias=True)
        #MultiThreadMemory init
        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)
        #Transformer
            #Encoder:3 x EncoderLayer(MultiHeadedAttention,PositionwiseFeedForward,SublayerConnection)
            #Decoder:3 x DecoderLayer
        self.model = self.make_model(tgt_vocab, self.cmn)
        #logit:Linear(in_features=512, out_features=761, bias=True)
        self.logit = nn.Linear(args.d_model, tgt_vocab)  #512,761
        #memory_matrix:torch.Size([2048, 512])
        # its dimension and the number of memory vectors N are set to 512 and 2048，and also randomly initialized
        # --------改为 31 x 512
        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.num_classes+1, args.cmm_dim))  # the numebr of cmm size:2048,the dimension of cmm dimension:512
        # nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)


    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, global_states, fc_feats, att_feats, att_masks):
        global_states, att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(global_states, att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, global_states, att_feats, att_masks=None, seq=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)    #（16，98，512）
        global_states = pack_wrapper(self.att_embed, global_states, att_masks) #(16,31,512)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)   #（16，98）

        # Memory querying and responding for visual features   dummy_memory_matrix：（16，2048，512）--->(16,31,512)
        # dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0), self.memory_matrix.size(1))
        dummy_memory_matrix = global_states
        #视觉responses       [16, 98, 512]-------做查询
        responses = self.cmn(att_feats, dummy_memory_matrix, dummy_memory_matrix)
        #([8, 98, 512]) = ([8, 98, 512])+([8, 98, 512])
        att_feats = att_feats + responses          #特征交互：相加？
        #改变fuse_feature后：[16, 98, 512]
        att_feats = self.fuse_feature(torch.cat((att_feats, responses), dim=2))
        # Memory querying and responding for visual features

        att_masks = att_masks.unsqueeze(-2)
        if seq is not None:
            seq = seq[:, :-1]
            seq_mask = (seq.data > 0)
            seq_mask[:, 0] += True

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
        else:
            seq_mask = None

        return global_states, att_feats, seq, att_masks, seq_mask

    def _forward(self, global_states, att_feats, seq, att_masks=None):
        # 1111111-------------------BaseCMN forward开始----------------------

        #之前att_feats:([16, 98, 2048])
        #patch features with 512 dimensions for each feature
        #得到之后：att_feats：([8, 98, 512])（得到视觉response输入encoder）
        global_states, att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(global_states, att_feats, att_masks, seq)

        #out：([8, X, 512]
        out = self.model(att_feats, seq, att_masks, seq_mask, memory_matrix=global_states)
        #outputs：([8, X, 761])
        #generator   ：logits全连接层
        outputs = F.log_softmax(self.logit(out), dim=-1)       #经过一个Linear，再做softmax，得到outputs(8, X, trg_vocab_size)

        return outputs

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask):
        if len(state) == 0:
            ys = it.unsqueeze(1)
            past = [fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model),
                    fc_feats_ph.new_zeros(self.num_layers * 2, fc_feats_ph.shape[0], 0, self.d_model)]
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
            past = state[1:]
        out, past = self.model.decode(memory, mask, ys, subsequent_mask(ys.size(1)).to(memory.device), past=past,
                                      memory_matrix=self.memory_matrix)
        return out[:, -1], [ys.unsqueeze(0)] + past
