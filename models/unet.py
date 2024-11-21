from torch.nn import functional as F
from torch.nn import MultiheadAttention
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=8)
        self.latent_proj = nn.Conv2d(128, embed_dim, kernel_size=1)  # Project latent to embed_dim
        self.text_proj = nn.Linear(embed_dim, embed_dim)  # Optional: Project text embedding to embed_dim

    def forward(self, latent, text_embedding):
        # Project latent features to embed_dim
        latent = self.latent_proj(latent)  # B, C, H, W -> B, embed_dim, H, W
        latent_flat = latent.flatten(2).permute(2, 0, 1)  # B, embed_dim, H*W -> (H*W, B, embed_dim)

        # Project text embeddings and expand for attention
        text_embedding = self.text_proj(text_embedding)  # Ensure it's embed_dim
        text_embedding = text_embedding.unsqueeze(0).repeat(latent_flat.size(0), 1, 1)  # (H*W, B, embed_dim)

        # Perform attention
        attn_output, _ = self.attention(latent_flat, text_embedding, text_embedding)

        # Reshape back to original latent dimensions
        return attn_output.permute(1, 2, 0).view_as(latent)  # (H*W, B, embed_dim) -> B, embed_dim, H, W

class UNet(nn.Module):
    def __init__(self, latent_dim, text_embed_dim):
        super(UNet, self).__init__()
        self.down1 = nn.Conv2d(latent_dim, 64, kernel_size=3, padding=1)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.attn = CrossAttention(embed_dim=text_embed_dim)  # Use text_embed_dim here
        self.reduce_channels = nn.Conv2d(text_embed_dim, 128, kernel_size=1)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.up2 = nn.ConvTranspose2d(64, latent_dim, kernel_size=3, padding=1)

    def forward(self, x, text_embedding):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.attn(x, text_embedding)
        x = self.reduce_channels(x)
        x = F.relu(self.up1(x))
        return self.up2(x)
