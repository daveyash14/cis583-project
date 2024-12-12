from torch.nn import functional as F
from torch.nn import MultiheadAttention
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(CrossAttention, self).__init__()
        self.attention = MultiheadAttention(embed_dim=embed_dim, num_heads=8)
        self.latent_proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)  # Project latent to embed_dim
        self.text_proj = nn.Linear(embed_dim, embed_dim)  # Project text embedding to embed_dim

    def forward(self, latent, text_embedding):
        # Project latent features to embed_dim
        latent = self.latent_proj(latent)  # B, C, H, W -> B, embed_dim, H, W
        latent_flat = latent.flatten(2).permute(2, 0, 1)  # B, embed_dim, H*W -> (H*W, B, embed_dim)

        # Project text embeddings to embed_dim
        text_embedding = self.text_proj(text_embedding)
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
        self.attn1 = CrossAttention(input_dim= 128, embed_dim=text_embed_dim)
        self.reduce_channels1 = nn.Conv2d(text_embed_dim, 128, kernel_size=1)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.attn2 = CrossAttention(input_dim=256, embed_dim=text_embed_dim)
        self.reduce_channels2 = nn.Conv2d(text_embed_dim, 256, kernel_size=1)
        self.down4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.attn3 = CrossAttention(input_dim=512, embed_dim=text_embed_dim)
        self.reduce_channels3 = nn.Conv2d(text_embed_dim, 512, kernel_size=1)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.attn4 = CrossAttention(input_dim=256, embed_dim=text_embed_dim)
        self.reduce_channels4 = nn.Conv2d(text_embed_dim, 256, kernel_size=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.attn5 = CrossAttention(input_dim=128, embed_dim=text_embed_dim)
        self.reduce_channels5 = nn.Conv2d(text_embed_dim, 128, kernel_size=1)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.up4 = nn.ConvTranspose2d(64, latent_dim, kernel_size=3, padding=1)

    def forward(self, x, text_embedding):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.attn1(x, text_embedding)
        x = self.reduce_channels1(x)
        x = F.relu(self.down3(x))
        x = self.attn2(x, text_embedding)
        x = self.reduce_channels2(x)
        x = F.relu(self.down4(x))
        x = self.attn3(x, text_embedding)
        x = self.reduce_channels3(x)
        x = F.relu(self.up1(x))
        x = self.attn4(x, text_embedding)
        x = self.reduce_channels4(x)
        x = F.relu(self.up2(x))
        x = self.attn5(x, text_embedding)
        x = self.reduce_channels5(x)
        x = F.relu(self.up3(x))
        return self.up4(x)