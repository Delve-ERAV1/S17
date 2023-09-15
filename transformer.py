import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import torch
import math
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Transformer
# =============================================================================
def attention(q, k, v, mask = None, dropout = None):
    scores = q.matmul(k.transpose(-2, -1))
    scores /= math.sqrt(q.shape[-1])
    
    #mask
    scores = scores if mask is None else scores.masked_fill(mask == 0, -1e3)
    
    scores = F.softmax(scores, dim = -1)
    scores = dropout(scores) if dropout is not None else scores
    output = scores.matmul(v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, out_dim, dropout=0.1):
        super().__init__()
        
#        self.q_linear = nn.Linear(out_dim, out_dim)
#        self.k_linear = nn.Linear(out_dim, out_dim)
#        self.v_linear = nn.Linear(out_dim, out_dim)
        self.linear = nn.Linear(out_dim, out_dim*3)

        self.n_heads = n_heads
        self.out_dim = out_dim
        self.out_dim_per_head = out_dim // n_heads
        self.out = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, t):
        return t.reshape(t.shape[0], -1, self.n_heads, self.out_dim_per_head)
    
    def forward(self, x, y=None, mask=None):
        #in decoder, y comes from encoder. In encoder, y=x
        y = x if y is None else y
        
        qkv = self.linear(x) # BS * SEQ_LEN * (3*EMBED_SIZE_L)
        q = qkv[:, :, :self.out_dim] # BS * SEQ_LEN * EMBED_SIZE_L
        k = qkv[:, :, self.out_dim:self.out_dim*2] # BS * SEQ_LEN * EMBED_SIZE_L
        v = qkv[:, :, self.out_dim*2:] # BS * SEQ_LEN * EMBED_SIZE_L
        
        #break into n_heads
        q, k, v = [self.split_heads(t) for t in (q,k,v)]  # BS * SEQ_LEN * HEAD * EMBED_SIZE_P_HEAD
        q, k, v = [t.transpose(1,2) for t in (q,k,v)]  # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        
        #n_heads => attention => merge the heads => mix information
        scores = attention(q, k, v, mask, self.dropout) # BS * HEAD * SEQ_LEN * EMBED_SIZE_P_HEAD
        scores = scores.transpose(1,2).contiguous().view(scores.shape[0], -1, self.out_dim) # BS * SEQ_LEN * EMBED_SIZE_L
        out = self.out(scores)  # BS * SEQ_LEN * EMBED_SIZE
        
        return out

class FeedForward(nn.Module):
    def __init__(self, inp_dim, inner_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(inp_dim, inner_dim)
        self.linear2 = nn.Linear(inner_dim, inp_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #inp => inner => relu => dropout => inner => inp
        return self.linear2(self.dropout(F.relu(self.linear1(x)))) 

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, inner_transformer_size, inner_ff_size, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, inner_transformer_size, dropout)
        self.ff = FeedForward(inner_transformer_size, inner_ff_size, dropout)
        self.norm1 = nn.LayerNorm(inner_transformer_size)
        self.norm2 = nn.LayerNorm(inner_transformer_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.mha(x2, mask=mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

class EncoderTransformer(nn.Module):
    def __init__(self, n_code, n_heads, embed_size, inner_ff_size, seq_len, n_embeddings=None, num_class=None, dropout=.1, isBERT=True):
        super().__init__()

        self.isBERT = isBERT

        if isBERT:
          self.embeddings = nn.Embedding(n_embeddings, embed_size)
        else:
          self.embeddings = nn.Conv2d(3, embed_size, 16, 16)
          self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embed_size), requires_grad=True)
          seq_len += 1 # Classification Token

        self.pe = nn.Parameter(torch.ones(1, seq_len, embed_size), requires_grad=True)

        encoders = []
        for i in range(n_code):
            encoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.encoders = nn.ModuleList(encoders)

        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, n_embeddings if isBERT else num_class, bias=False)
        self.flatten = nn.Flatten(2, 3)

    def forward(self, x):

        x = self.embeddings(x)

        if not self.isBERT:
          x = self.flatten(x).permute(0, 2, 1)
          class_token = self.class_embedding.expand(x.shape[0], -1, -1)
          x = torch.cat((class_token, x), dim=1)

        x = x + self.pe
        for encoder in self.encoders:
            x = encoder(x)

        x = self.norm(x)
        x = self.linear(x if self.isBERT else x[:, 0])
        return x
    

class DecoderTransformer(nn.Module):
    def __init__(self, n_layers, n_heads, embed_size, inner_ff_size, seq_len, vocab_size, dropout=0.1):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(seq_len, embed_size)

        decoders = []
        for i in range(n_layers):
            decoders += [EncoderLayer(n_heads, embed_size, inner_ff_size, dropout)]
        self.decoders = nn.ModuleList(decoders)

        self.norm = nn.LayerNorm(embed_size)
        self.linear = nn.Linear(embed_size, vocab_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, input):
      B, T = input.shape
      token_embed = self.embeddings(input)
      pos_embed = self.position_embedding_table(torch.arange(T, device=DEVICE))

      x = token_embed + pos_embed

      for decoder in self.decoders:
        x = decoder(x, mask=self.tril[:T, :T])

      logits = self.linear(x)
      return(logits)