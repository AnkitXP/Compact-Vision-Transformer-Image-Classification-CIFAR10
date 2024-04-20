import torch
from torch import nn
import torch.nn.functional as F
import sys

# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels, patch_size, embedding_dim):
#         super().__init__()
#         self.patch_size = patch_size
#         self.patcher = nn.Conv2d(in_channels = in_channels,
#                                  out_channels = embedding_dim,
#                                  kernel_size = patch_size,
#                                  stride = patch_size,
#                                  padding = 0)
        
#         self.flatten = nn.Flatten(start_dim = 2, end_dim = 3)

#     def forward(self, x):
#         image_resolution = x.shape[-1]
#         assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

#         x_patched = self.patcher(x)
#         x_flattened = self.flatten(x_patched) 
        
#         # 6. Make sure the output shape has the right order 
#         return x_flattened.permute(0, 2, 1)
    

# class MultiheadSelfAttentionBlock(nn.Module):
#     """Creates a multi-head self-attention block ("MSA block" for short).
#     """
#     # 2. Initialize the class with hyperparameters from Table 1
#     def __init__(self,
#                  embedding_dim:int=768, 
#                  num_heads:int=12, 
#                  attn_dropout:float=0): 
#         super().__init__()
        
#         # 3. Create the Norm layer (LN)
#         self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
#         # 4. Create the Multi-Head Attention (MSA) layer
#         self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
#                                                     num_heads=num_heads,
#                                                     dropout=attn_dropout,
#                                                     batch_first=True) 
        
#     # 5. Create a forward() method to pass the data throguh the layers
#     def forward(self, x):
#         x = self.layer_norm(x)
#         attn_output, _ = self.multihead_attn(query=x, 
#                                              key=x, 
#                                              value=x, 
#                                              need_weights=False) 
#         return attn_output
    

# class MLPBlock(nn.Module):
#     """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
#     # 2. Initialize the class with hyperparameters from Table 1 and Table 3
#     def __init__(self,
#                  embedding_dim:int=768, 
#                  mlp_size:int=3072, 
#                  dropout:float=0.1): 
#         super().__init__()
        
#         # 3. Create the Norm layer (LN)
#         self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
#         # 4. Create the Multilayer perceptron (MLP) layer(s)
#         self.mlp = nn.Sequential(
#             nn.Linear(in_features=embedding_dim,
#                       out_features=mlp_size),
#             nn.GELU(), 
#             nn.Dropout(p=dropout),
#             nn.Linear(in_features=mlp_size, 
#                       out_features=embedding_dim), 
#             nn.Dropout(p=dropout) 
#         )
    
#     # 5. Create a forward() method to pass the data throguh the layers
#     def forward(self, x):
#         x = self.layer_norm(x)
#         x = self.mlp(x)
#         return x
    
# class TransformerEncoderBlock(nn.Module):
#     """Creates a Transformer Encoder block."""
#     # 2. Initialize the class with hyperparameters from Table 1 and Table 3
#     def __init__(self,
#                  embedding_dim:int=768, 
#                  num_heads:int=12, 
#                  mlp_size:int=3072,
#                  mlp_dropout:float=0.1,
#                  attn_dropout:float=0): 
#         super().__init__()

#         # 3. Create MSA block (equation 2)
#         self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
#                                                      num_heads=num_heads,
#                                                      attn_dropout=attn_dropout)
        
#         # 4. Create MLP block (equation 3)
#         self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
#                                    mlp_size=mlp_size,
#                                    dropout=mlp_dropout)
        
#     # 5. Create a forward() method  
#     def forward(self, x):
        
#         # 6. Create residual connection for MSA block (add the input to the output)
#         x =  self.msa_block(x) + x 
        
#         # 7. Create residual connection for MLP block (add the input to the output)
#         x = self.mlp_block(x) + x 
        
#         return x
    
# class ViT(nn.Module):
#     """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
#     # 2. Initialize the class with hyperparameters from Table 1 and Table 3
#     def __init__(self, configs):
        
#         super().__init__() 
#         self.img_size = configs.img_size 
#         self.in_channels = configs.in_channels 
#         self.patch_size = configs.patch_size 
#         self.num_transformer_layers = configs.num_encoders 
#         self.embedding_dim = (self.patch_size ** 2) * self.in_channels 
#         self.mlp_size = configs.mlp_size 
#         self.num_heads = configs.num_heads 
#         self.attn_dropout = 0.0
#         self.mlp_dropout = configs.dropout_value
#         self.embedding_dropout = configs.dropout_value
#         self.num_classes = configs.num_classes

        
#         # 3. Make the image size is divisble by the patch size 
#         assert self.img_size % self.patch_size == 0, f"Image size must be divisible by patch size, image size: {self.img_size}, patch size: {self.patch_size}."
        
#         # 4. Calculate number of patches (height * width/patch^2)
#         self.num_patches = (self.img_size * self.img_size) // self.patch_size**2
                 
#         # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
#         self.class_embedding = nn.Parameter(data=torch.randn(1, 1, self.embedding_dim),
#                                             requires_grad=True)
        
#         # 6. Create learnable position embedding
#         self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, self.embedding_dim),
#                                                requires_grad=True)
                
#         # 7. Create embedding dropout value
#         self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
        
#         # 8. Create patch embedding layer
#         self.patch_embedding = PatchEmbedding(in_channels=self.in_channels,
#                                               patch_size=self.patch_size,
#                                               embedding_dim=self.embedding_dim)
        
#         # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential()) 
#         # Note: The "*" means "all"
#         self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=self.embedding_dim,
#                                                                             num_heads=self.num_heads,
#                                                                             mlp_size=self.mlp_size,
#                                                                             mlp_dropout=self.mlp_dropout) for _ in range(self.num_transformer_layers)])
       
#         # 10. Create classifier head
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(normalized_shape=self.embedding_dim),
#             nn.Linear(in_features=self.embedding_dim, 
#                       out_features=self.num_classes)
#         )
    
#     # 11. Create a forward() method
#     def forward(self, x):
        
#         # 12. Get batch size
#         batch_size = x.shape[0]
        
#         # 13. Create class token embedding and expand it to match the batch size (equation 1)
#         class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

#         # 14. Create patch embedding (equation 1)
#         x = self.patch_embedding(x)

#         # 15. Concat class embedding and patch embedding (equation 1)
#         x = torch.cat((class_token, x), dim=1)

#         # 16. Add position embedding to patch embedding (equation 1) 
#         x = self.position_embedding + x

#         # 17. Run embedding dropout (Appendix B.1)
#         x = self.embedding_dropout(x)

#         # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
#         x = self.transformer_encoder(x)

#         # 19. Put 0 index logit through classifier (equation 4)
#         x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

#         return x

# ---------------------------------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return x + self.gamma * self.residual(x)
    

class LayerNormChannels(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x
    

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, head_channels, shape):
        super().__init__()
        self.heads = out_channels // head_channels
        self.head_channels = head_channels
        self.scale = head_channels**-0.5
        
        self.to_keys = nn.Conv2d(in_channels, out_channels, 1)
        self.to_queries = nn.Conv2d(in_channels, out_channels, 1)
        self.to_values = nn.Conv2d(in_channels, out_channels, 1)
        self.unifyheads = nn.Conv2d(out_channels, out_channels, 1)
        
        height, width = shape
        self.pos_enc = nn.Parameter(torch.Tensor(self.heads, (2 * height - 1) * (2 * width - 1)))
        self.register_buffer("relative_indices", self.get_indices(height, width))
    
    def forward(self, x):
        b, _, h, w = x.shape
        
        keys = self.to_keys(x).view(b, self.heads, self.head_channels, -1)
        values = self.to_values(x).view(b, self.heads, self.head_channels, -1)
        queries = self.to_queries(x).view(b, self.heads, self.head_channels, -1)
        
        att = keys.transpose(-2, -1) @ queries
        
        indices = self.relative_indices.expand(self.heads, -1)
        rel_pos_enc = self.pos_enc.gather(-1, indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (h * w, h * w))
        
        att = att * self.scale + rel_pos_enc
        att = F.softmax(att, dim=-2)
        
        out = values @ att
        out = out.view(b, -1, h, w)
        out = self.unifyheads(out)
        return out
    
    @staticmethod
    def get_indices(h, w):
        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)
        
        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()
        
        return indices
    

class FeedForward(nn.Sequential):
    def __init__(self, in_channels, out_channels, mult=4):
        hidden_channels = in_channels * mult
        super().__init__(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1)   
        )

class TransformerBlock(nn.Sequential):
    def __init__(self, channels, head_channels, shape, p_drop=0.):
        super().__init__(
            Residual(
                LayerNormChannels(channels),
                SelfAttention2d(channels, channels, head_channels, shape),
                nn.Dropout(p_drop)
            ),
            Residual(
                LayerNormChannels(channels),
                FeedForward(channels, channels),
                nn.Dropout(p_drop)
            )
        )

class TransformerStack(nn.Sequential):
    def __init__(self, num_blocks, channels, head_channels, shape, p_drop=0.):
        layers = [TransformerBlock(channels, head_channels, shape, p_drop) for _ in range(num_blocks)]
        super().__init__(*layers)

class ToPatches(nn.Sequential):
    def __init__(self, in_channels, channels, patch_size, hidden_channels=32):
        super().__init__(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, patch_size, stride=patch_size)
        )

class AddPositionEmbedding(nn.Module):
    def __init__(self, channels, shape):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.Tensor(channels, *shape))
    
    def forward(self, x):
        return x + self.pos_embedding
    
class ToEmbedding(nn.Sequential):
    def __init__(self, in_channels, channels, patch_size, shape, p_drop=0.):
        super().__init__(
            ToPatches(in_channels, channels, patch_size),
            AddPositionEmbedding(channels, shape),
            nn.Dropout(p_drop)
        )

class Head(nn.Sequential):
    def __init__(self, in_channels, classes, p_drop=0.):
        super().__init__(
            LayerNormChannels(in_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(in_channels, classes)
        )

class RelViT(nn.Sequential):
    def __init__(self, configs):
        classes=10 
        image_size =32
        channels=256
        head_channels=32
        num_blocks=8
        patch_size=2
        in_channels=3
        emb_p_drop=0.
        trans_p_drop=0.
        head_p_drop=0.

        reduced_size = image_size // patch_size
        shape = (reduced_size, reduced_size)
        
        super().__init__(
            ToEmbedding(in_channels, channels, patch_size, shape, emb_p_drop),
            TransformerStack(num_blocks, channels, head_channels, shape, trans_p_drop),
            Head(channels, classes, head_p_drop)
        )
        
    def forward(self, x):
        # Pass input through the layers sequentially
        return super().forward(x)