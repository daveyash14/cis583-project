import torch
import torch.nn as nn

class Diffusion(nn.Module):
    def __init__(self, noise_schedule, device="cpu"):
        super(Diffusion, self).__init__()
        self.noise_schedule = noise_schedule.to(device)
        self.device = device

    def forward_diffusion(self, x, t):
        """Apply noise to the latent representation."""
        alpha_t = self.noise_schedule[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        noise = torch.randn_like(x, device=self.device)
        return x * torch.sqrt(1 - alpha_t) + noise * torch.sqrt(alpha_t)

    def reverse_diffusion(self, x, t, predicted_noise):
        """Remove noise from the latent representation."""
        alpha_t = self.noise_schedule[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return (x - predicted_noise * torch.sqrt(alpha_t)) / torch.sqrt(1 - alpha_t)
