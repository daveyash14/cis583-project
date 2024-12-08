import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, latent_dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(64, latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential(
            # nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(latent_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1),
            # nn.Tanh() # For flickr
            nn.Sigmoid() # For mnist
        )

    def forward(self, x):
        return self.deconv(x)
