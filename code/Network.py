import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

class CompactVisionTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()

        image_size = configs.img_size
        patch_size = configs.patch_size
        num_classes = configs.num_classes
        dim = configs.dim
        depth = configs.depth
        heads = configs.num_heads
        in_channels = configs.in_channels
        dim_head = configs.dim_head
        dropout = configs.dropout
        emb_dropout = configs.embed_dropout
        scale_dim = configs.scale_dim

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size provided.'
        
        num_patches = (image_size // patch_size) ** 2
        self.to_patch_embedding = ConvEmbed(in_channels, dim)
        num_patches = self.to_patch_embedding.sequence_length()

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad = True)
        
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.pool = nn.Linear(dim, 1)

        self.norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, num_classes)
        )

        self.apply(self.init_weight)

    def forward(self, img):

        x = self.to_patch_embedding(img)
        b, n, d = x.shape

        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = self.norm(x)

        g = self.pool(x)

        xl = F.softmax(g, dim=1)

        x = torch.matmul(xl.transpose(-1, -2), x).squeeze(-2)

        return self.mlp_head(x)
    
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class ConvEmbed(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, pool_kernel_size=2, pool_stride=2,
                 pool_padding=0):
        super(ConvEmbed, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                    padding=padding, bias=False),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
            )
        
        self.flatten = nn.Flatten(2, 3)

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=32, width=32):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = x.transpose(-2, -1)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNormChannels(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                LayerNormChannels(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    
class LayerNormChannels(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, d, h = *x.shape, self.heads
        q = self.to_q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.to_k(x).view(b, n, h, -1).transpose(1, 2)
        v = self.to_v(x).view(b, n, h, -1).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(b, n, -1)

        out = self.to_out(out)

        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
