import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
        timesteps: [N] dimensional tensor of int.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
        return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def timesteps_to_tensor(ts: int or list[int], batch_size):
    if isinstance(ts, list):
        assert batch_size % len(ts) == 0, "batch_size must be divisible by length of timesteps list"
    
    if isinstance(ts, int):
        return ts * torch.ones(batch_size, device=ts.device)
    else:
        mini_batch_size = batch_size // len(ts)
        return torch.cat([ts[i] * torch.ones(mini_batch_size, device=ts.device) for i in range(len(ts))])


class Toy_Discriminator(nn.Module):
    def __init__(self, args):
        super(Toy_Discriminator, self).__init__()
        self.time_embed_dim = args.time_embed_dim # 128
        dim = 512
        out_dim = 1
        
        self.t_module = nn.Sequential(nn.Linear(self.time_embed_dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim),)
        self.x_module = ResNet_FC(2, dim, num_res_blocks=2)
        self.out_module = nn.Sequential(nn.Linear(dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, out_dim),)
        
    def forward(self, x_t, t, x_tp1):
        t = timesteps_to_tensor(t, batch_size=x_t.shape[0]).to(x_t.device)
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        
        
        x_in = torch.cat([x_t, x_tp1], dim=1)
        # sys.exit()
        x_out = self.x_module(x_in)
        out   = self.out_module(x_out+t_out)

        return out
    
    
class Toy_Generator(nn.Module):
    def __init__(self, args, direction=None):
        super(Toy_Generator, self).__init__()
        self.direction = direction

        self.time_embed_dim = args.time_embed_dim
        self.z_dim = args.nz
        dim = 512
        out_dim = 1

        self.t_module = nn.Sequential(nn.Linear(self.time_embed_dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim),)
        self.z_module = nn.Sequential(PixelNorm(), nn.Linear(self.z_dim, dim), nn.LeakyReLU(0.2))
        self.x_module = ResNet_FC(1, dim, num_res_blocks=3)
        self.out_module = nn.Sequential(nn.Linear(2*dim, dim), nn.LeakyReLU(0.2), nn.Linear(dim, out_dim),)

    def forward(self, x, t: int or list[int], z):
        t = timesteps_to_tensor(t, batch_size=x.shape[0]).to(x.device)
        t_emb = timestep_embedding(t, self.time_embed_dim)
        z = self.z_module(z)

        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        x_out += t_out
        x_out = torch.cat([x_out, z], dim=1)
        out   = self.out_module(x_out)

        return out
    
class ResNet_FC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.map=nn.Linear(data_dim, hidden_dim)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])

    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        h=self.map(x)
        for res_block in self.res_blocks:
            h = (h + res_block(h)) / np.sqrt(2)
        return h