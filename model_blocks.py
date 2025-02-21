from typing import Dict, Tuple, Optional
import torch.nn as nn
import torch
import torch.nn.functional as F

### Model Blocks ###

class FeedForward(nn.Module):
  def __init__(self, d_model: int, ff_hidden_ratio: int=4):
    super().__init__()
    self.hidden_dim = d_model * ff_hidden_ratio
    self.d_model = d_model

    self.net = nn.Sequential(
        nn.Linear(d_model, self.hidden_dim),
        nn.GELU(),
        nn.Linear(self.hidden_dim, d_model)
    )

  def forward(self, x):
    return self.net(x)
    

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int=1, causal_mask: bool=True):
        super().__init__()
        assert d_model % n_heads == 0 # d_model must be divisible by n_heads

        self.n_heads = n_heads
        self.causal_mask = causal_mask
        self.d_model = d_model
        self.d_k = d_model // n_heads

        # create multi heads attention projections
        self.Wq = nn.Linear(d_model, self.d_k * n_heads)
        self.Wk = nn.Linear(d_model, self.d_k * n_heads)
        self.Wv = nn.Linear(d_model, self.d_k * n_heads)

        # Final projection back to d_model
        self.Wo = nn.Linear(self.d_k * n_heads, d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, dict]:
        batch_size, seq_len, _ = x.size()

        Q = self.Wq(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, device=x.device))

        if attention_mask is not None:
            # Convert to boolean and expand dims for broadcasting
            # attention_mask is expected to be (batch_size, seq_len) where 1 means keep, 0 means mask
            attention_mask = attention_mask.bool()  # Convert to boolean
            attention_mask = ~attention_mask  # Invert because masked_fill masks where True
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(attention_mask, float('-inf'))

        # add masking here for all the heads
        if self.causal_mask:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        O = self.Wo(attn_output)

        return O, {'attn_weights' : attn_weights} 
    

class DecoderBlock(nn.Module):
    def __init__(self, decoder_d_model: int=512, n_heads:int=1, ff_hidden_ratio: int=4, dropout: float=0.2):
        super().__init__()
        self.decoder_d_model = decoder_d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.attn1 = SelfAttention(d_model=decoder_d_model, n_heads=n_heads, causal_mask=True)
        self.norm1 = nn.LayerNorm(decoder_d_model)
        self.norm2 = nn.LayerNorm(decoder_d_model)
        self.ff = FeedForward(decoder_d_model, ff_hidden_ratio)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        x_att1, self_attn_dict = self.attn1(x, attention_mask)
        x_att1 = self.dropout(x_att1)
        x_norm1 = self.norm1(x + x_att1)
        
        x_ff = self.ff(x_norm1)
        x_ff = self.dropout(x_ff)
        x_out = self.norm2(x_norm1 + x_ff)

        attention_info = {
            "self_attention": self_attn_dict["attn_weights"]
        }

        return x_out, attention_info
    