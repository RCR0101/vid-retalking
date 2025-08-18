import torch
from torch import nn
import math

from einops import rearrange

import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class DualPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.normx = nn.LayerNorm(dim)
        self.normy = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, y, **kwargs):
        return self.fn(self.normx(x), self.normy(y), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        # qk = self.to_qk(x).chunk(2, dim = -1) #
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.heads) # q,k from the zero feature
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h = self.heads) # v from the reference features
        v = rearrange(self.to_v(y), 'b n (h d) -> b h n d', h = self.heads) 

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) for better temporal understanding"""
    def __init__(self, dim, max_position=512, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

def rotate_half(x):
    """Helper function for RoPE"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    """Apply rotary position embedding to q and k"""
    q_embed = (q * freqs_cos) + (rotate_half(q) * freqs_sin)
    k_embed = (k * freqs_cos) + (rotate_half(k) * freqs_sin)
    return q_embed, k_embed


class TemporalAwareCrossAttention(nn.Module):
    """Enhanced cross-attention with temporal awareness, learnable positional embeddings, and RoPE"""
    def __init__(self, dim, heads=16, dim_head=64, dropout=0., max_temporal_len=32):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.max_temporal_len = max_temporal_len

        self.attend = nn.Softmax(dim=-1)
        
        # Linear projections for Q, K, V
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        # Learnable temporal positional embeddings
        self.temporal_pos_emb = nn.Parameter(torch.randn(max_temporal_len, dim))
        
        # RoPE for temporal understanding
        self.rope = RotaryPositionalEmbedding(dim_head)
        
        # Temporal fusion weights
        self.temporal_gate = nn.Linear(dim, 1)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y, temporal_mask=None):
        """
        Args:
            x: query features [batch, temporal_len, spatial_tokens, dim]
            y: key-value features [batch, temporal_len, spatial_tokens, dim] 
            temporal_mask: optional temporal masking [batch, temporal_len, temporal_len]
        """
        batch_size, temporal_len, spatial_tokens, dim = x.shape
        
        # Add learnable temporal positional embeddings
        if temporal_len <= self.max_temporal_len:
            temp_pos = self.temporal_pos_emb[:temporal_len]  # [temporal_len, dim]
            x = x + temp_pos.unsqueeze(0).unsqueeze(2)  # [batch, temporal_len, 1, dim]
            y = y + temp_pos.unsqueeze(0).unsqueeze(2)
        
        # Reshape for attention: [batch, temporal_len * spatial_tokens, dim]
        x_flat = x.reshape(batch_size, temporal_len * spatial_tokens, dim)
        y_flat = y.reshape(batch_size, temporal_len * spatial_tokens, dim)
        
        # Generate Q, K, V
        q = rearrange(self.to_q(x_flat), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x_flat), 'b n (h d) -> b h n d', h=self.heads) 
        v = rearrange(self.to_v(y_flat), 'b n (h d) -> b h n d', h=self.heads)
        
        # Apply RoPE to Q and K for temporal understanding
        freqs = self.rope(temporal_len * spatial_tokens, x.device)
        freqs_cos = freqs.cos()
        freqs_sin = freqs.sin()
        
        # Broadcast freqs for multi-head attention
        freqs_cos = freqs_cos.unsqueeze(1).expand(-1, self.heads, -1, -1)
        freqs_sin = freqs_sin.unsqueeze(1).expand(-1, self.heads, -1, -1)
        
        q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply temporal mask if provided
        if temporal_mask is not None:
            # Expand temporal mask to spatial dimensions
            expanded_mask = temporal_mask.unsqueeze(2).unsqueeze(4)  # [batch, temp, 1, temp, 1]
            expanded_mask = expanded_mask.expand(-1, -1, spatial_tokens, -1, spatial_tokens)
            expanded_mask = expanded_mask.reshape(batch_size, 1, temporal_len * spatial_tokens, temporal_len * spatial_tokens)
            dots = dots.masked_fill(expanded_mask == 0, float('-inf'))
        
        attn = self.attend(dots)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Apply output projection
        out = self.to_out(out)
        
        # Reshape back to temporal structure
        out = out.reshape(batch_size, temporal_len, spatial_tokens, dim)
        
        # Temporal gating for adaptive temporal fusion
        temporal_weights = torch.sigmoid(self.temporal_gate(out))  # [batch, temp, spatial, 1]
        out = out * temporal_weights
        
        return out


class EnhancedAttention(nn.Module):
    """Enhanced version of the original Attention with 16+ heads"""
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(y), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                DualPreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))


    def forward(self, x, y): # x is the cropped, y is the foreign reference
        bs,c,h,w = x.size()

        # img to embedding
        x = x.view(bs,c,-1).permute(0,2,1)
        y = y.view(bs,c,-1).permute(0,2,1)

        for attn, ff in self.layers:
            x = attn(x, y) + x
            x = ff(x) + x

        x = x.view(bs,h,w,c).permute(0,3,1,2)
        return x


class TemporalTransformer(nn.Module):
    """Enhanced Transformer with temporal-aware cross-attention support"""
    def __init__(self, dim, depth, heads=16, dim_head=64, mlp_dim=None, dropout=0., 
                 use_temporal_attention=True, max_temporal_len=32):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = dim * 4
            
        self.use_temporal_attention = use_temporal_attention
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            if use_temporal_attention:
                # Use temporal-aware cross-attention
                attention_layer = DualPreNorm(dim, 
                    TemporalAwareCrossAttention(dim, heads=heads, dim_head=dim_head, 
                                              dropout=dropout, max_temporal_len=max_temporal_len))
            else:
                # Use enhanced attention with more heads
                attention_layer = DualPreNorm(dim, 
                    EnhancedAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
                
            self.layers.append(nn.ModuleList([
                attention_layer,
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, y, temporal_mask=None):
        """
        Args:
            x: query features [batch, channels, height, width] or [batch, temporal, channels, height, width]
            y: key-value features [batch, channels, height, width] or [batch, temporal, channels, height, width]
            temporal_mask: optional temporal mask for temporal attention
        """
        if self.use_temporal_attention and len(x.shape) == 5:
            # Temporal mode: [batch, temporal, channels, height, width]
            batch_size, temporal_len, channels, height, width = x.shape
            
            # Reshape to [batch, temporal, spatial_tokens, dim]
            x_temp = x.reshape(batch_size, temporal_len, channels, height * width).permute(0, 1, 3, 2)
            y_temp = y.reshape(batch_size, temporal_len, channels, height * width).permute(0, 1, 3, 2)
            
            for attn, ff in self.layers:
                if hasattr(attn.fn, 'forward') and 'temporal_mask' in attn.fn.forward.__code__.co_varnames:
                    # Temporal-aware attention
                    x_temp = attn(x_temp, y_temp, temporal_mask=temporal_mask) + x_temp
                else:
                    # Fallback to standard attention
                    x_flat = x_temp.reshape(batch_size, temporal_len * height * width, channels)
                    y_flat = y_temp.reshape(batch_size, temporal_len * height * width, channels)
                    x_flat = attn(x_flat, y_flat) + x_flat
                    x_temp = x_flat.reshape(batch_size, temporal_len, height * width, channels)
                
                # Apply feedforward
                x_temp_ff = x_temp.reshape(batch_size * temporal_len, height * width, channels)
                x_temp_ff = ff(x_temp_ff) + x_temp_ff
                x_temp = x_temp_ff.reshape(batch_size, temporal_len, height * width, channels)
            
            # Reshape back to [batch, temporal, channels, height, width]
            x = x_temp.permute(0, 1, 3, 2).reshape(batch_size, temporal_len, channels, height, width)
            
        else:
            # Standard spatial mode: [batch, channels, height, width]
            batch_size, channels, height, width = x.shape
            
            # img to embedding: [batch, spatial_tokens, channels]
            x = x.view(batch_size, channels, -1).permute(0, 2, 1)
            y = y.view(batch_size, channels, -1).permute(0, 2, 1)
            
            for attn, ff in self.layers:
                x = attn(x, y) + x
                x = ff(x) + x
            
            # Back to image format
            x = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
            
        return x


class EnhancedTransformer(nn.Module):
    """Drop-in replacement for original Transformer with 16+ heads"""
    def __init__(self, dim, depth, heads=16, dim_head=64, mlp_dim=None, dropout=0.):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = dim * 4
            
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                DualPreNorm(dim, EnhancedAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, y):
        batch_size, channels, height, width = x.size()

        # img to embedding
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)
        y = y.view(batch_size, channels, -1).permute(0, 2, 1)

        for attn, ff in self.layers:
            x = attn(x, y) + x
            x = ff(x) + x

        x = x.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return x

class RETURNX(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, y): # x is the cropped, y is the foreign reference 
        return x