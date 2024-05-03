import random
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

import lightning as pl
import os
import json


class BaseDataModule(pl.LightningDataModule):
  def __init__(self, batch_size=32, split=0.8, *args, **kwargs):
    super().__init__()
    self.ds_X, self.ds_Y = self.get_dataset(*args, **kwargs)
    self.split = int(self.ds_X.shape[0]*split)
    self.batch_size = batch_size

  def train_dataloader(self):
    ds_X_train, ds_Y_train = self.ds_X[0:self.split], self.ds_Y[0:self.split]
    return torch.utils.data.DataLoader(list(zip(ds_X_train, ds_Y_train)), batch_size=self.batch_size)

  def val_dataloader(self):
    ds_X_test, ds_Y_test = self.ds_X[self.split:], self.ds_Y[self.split:]
    return torch.utils.data.DataLoader(list(zip(ds_X_test, ds_Y_test)), batch_size=self.batch_size)

class ReverseDataModule(BaseDataModule):
  def get_dataset(self, cnt=10000, seq_len=6):
    ds = np.random.randint(0, 10, size=(cnt, seq_len))
    return ds, ds[:, ::-1].ravel().reshape(cnt, seq_len)
  
# dataset idea from https://github.com/karpathy/minGPT/blob/master/play_math.ipynb
class AdditionDataModule(BaseDataModule):
  def get_dataset(self):
    ret = []
    for i in range(100):
      for j in range(100):
        s = i+j
        ret.append([i//10, i%10, j//10, j%10, s//100, (s//10)%10, s%10])
    ds = np.array(ret)
    return ds[:, 0:6], np.copy(ds[:, 1:])    

# this is the hardest to learn and requires 4 layers
class ParityDataModule(BaseDataModule):
  def get_dataset(self, seq_len=10):
    ds_X, ds_Y = [], []
    for i in range(2**seq_len):
      x = [int(x) for x in list(bin(i)[2:].rjust(seq_len, '0'))]
      ds_X.append(x)
      ds_Y.append((np.cumsum(x)%2).tolist())
    return np.array(ds_X), np.array(ds_Y)
  
class WikipediaDataModule(BaseDataModule):
  def get_dataset(self, seq_len=50):
    global enwik8
    if 'enwik8' not in globals():
      import requests
      enwik8_zipped = requests.get("https://data.deepai.org/enwik8.zip").content
      from zipfile import ZipFile
      import io
      enwik8 = ZipFile(io.BytesIO(enwik8_zipped)).read('enwik8')
    en = np.frombuffer(enwik8, dtype=np.uint8).astype(np.int)
    en = en[0:-seq_len+1]
    en[en>127] = 127
    return en[0:-1].reshape(-1, seq_len), en[1:].reshape(-1, seq_len)
  
def attention(queries, keys, values):
  d = queries.shape[-1]
  scores = torch.matmul(queries, keys.transpose(-2,-1))/math.sqrt(d)
  attention_weights = F.softmax(scores, dim=-1)
  return torch.matmul(attention_weights, values)

class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.embed_dim, self.num_heads = embed_dim, num_heads
    assert embed_dim % num_heads == 0
    self.projection_dim = embed_dim // num_heads
    
    self.W_q = nn.Linear(embed_dim, embed_dim)
    self.W_k = nn.Linear(embed_dim, embed_dim)
    self.W_v = nn.Linear(embed_dim, embed_dim)
    self.W_o = nn.Linear(embed_dim, embed_dim)

  def transpose(self, x):
    x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.projection_dim)
    return x.permute(0, 2, 1, 3)
  
  def transpose_output(self, x):
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], self.embed_dim)
    
  def forward(self, q, k, v):
    q = self.transpose(self.W_q(q))
    k = self.transpose(self.W_k(k))
    v = self.transpose(self.W_v(v))
    output = attention(q, k, v)
    return self.W_o(self.transpose_output(output))
  
class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
    super(TransformerBlock, self).__init__()
    self.att = MultiHeadAttention(embed_dim, num_heads)
    self.ffn = nn.Sequential(
      nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim)
    )
    self.layernorm1 = nn.LayerNorm(embed_dim)
    self.layernorm2 = nn.LayerNorm(embed_dim)
    self.dropout = nn.Dropout(rate)
    
  def forward(self, x):
    x = self.layernorm1(x + self.dropout(self.att(x, x, x)))
    x = self.layernorm2(x + self.dropout(self.ffn(x)))
    return x
  
class TokenAndPositionEmbedding(nn.Module):
  def __init__(self, maxlen, vocab_size, embed_dim):
    super(TokenAndPositionEmbedding, self).__init__()
    self.token_emb = nn.Embedding(vocab_size, embed_dim)
    self.pos_emb = nn.Embedding(maxlen, embed_dim)
  def forward(self, x):
    pos = torch.arange(0, x.size(1), dtype=torch.int32, device=x.device)
    return self.token_emb(x) + self.pos_emb(pos).view(1, x.size(1), -1)

class LittleTransformer(pl.LightningModule):
  def __init__(self, seq_len=6, max_value=10, layer_count=2, embed_dim=128, num_heads=4, ff_dim=32):
    super().__init__()
    self.max_value = max_value
    self.model = nn.Sequential(
      TokenAndPositionEmbedding(seq_len, max_value, embed_dim),
      *[TransformerBlock(embed_dim, num_heads, ff_dim) for x in range(layer_count)],
      nn.Linear(embed_dim, max_value),
      nn.LogSoftmax(dim=-1))
    
  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    output = self.model(x)
    loss = F.nll_loss(output.view(-1, self.max_value), y.view(-1))
    self.log("train_loss", loss)
    return loss
  
  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    pred = self.model(x).argmax(dim=2)
    val_accuracy = (pred == y).type(torch.float).mean()
    self.log("val_accuracy", val_accuracy, prog_bar=True)
  
  def configure_optimizers(self):
    if self.device.type == 'cuda':
      # import apex
      # return apex.optimizers.FusedAdam(self.parameters(), lr=3e-4)
      return torch.optim.Adam(self.parameters(), lr=3e-4)
    else:
      return torch.optim.Adam(self.parameters(), lr=3e-4)
    

def get_model(seq_len=6,block_size=64, max_epochs=1, max_value=10, num_layers=2, embed_dim=128, n_head=4, ff_dim=32):
    model = LittleTransformer(seq_len, max_value, num_layers, embed_dim, n_head, ff_dim)
    trainer = pl.Trainer(enable_progress_bar=True, max_epochs=max_epochs)
    data = AdditionDataModule(block_size=block_size)
    trainer.fit(model, data)
    shape = [1, 6]
    x = torch.zeros(shape, dtype=torch.long)
    x = x.reshape(shape)
    return model, shape, x
