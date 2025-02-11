# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np

import os
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from datasets_prep.lsun import LSUN
from datasets_prep.lmdb_datasets import LMDBDataset
from torch.utils.data import Subset


from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

from util.data_process import getCleanData, getMixedData
from util.args_parser import args_parser
from util.utility import copy_source, broadcast_params, q_sample_pairs, sample_posterior, sample_from_model, select_phi
from util.diffusion_coefficients import get_time_schedule, get_sigma_schedule
from diffusers.models import AutoencoderKL

import sys
from datetime import datetime


class Diffusion_Coefficients():
    def __init__(self, args, device):
                
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1
        
        self.a_s_cum = self.a_s_cum.to(device)
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)
    
#%% posterior sampling
class Posterior_Coefficients():
    def __init__(self, args, device):
        
        _, _, self.betas = get_sigma_schedule(args, device=device)
        
        #we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
                                    (torch.tensor([1.], dtype=torch.float32,device=device), self.alphas_cumprod[:-1]), 0
                                        )               
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)
        
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

#%%
def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large, Discriminator_tiny
    from score_sde.models.ncsnpp_generator_adagn import NCSNpp
    from EMA import EMA
    
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = args.batch_size
    
    nz = args.nz #latent dimension
    
    if args.perturb_dataset == 'none':
        dataset = getCleanData(args.dataset, image_size=args.image_size)
    else:
        dataset = getMixedData(args.dataset, args.perturb_dataset, percentage = args.perturb_percent, image_size=args.image_size, shuffle=args.shuffle)
    num_samples = len(dataset)
    num_chunks = num_samples // batch_size
    num_spare_data = num_samples % batch_size
    if num_spare_data:
        subset_indices = list(range(num_samples - num_spare_data))
        dataset = Subset(dataset, subset_indices)
        print(f'There are {num_samples % batch_size} spare data!')
    print("Finish loading dataset")
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               drop_last = True)
    

    

    
    exp = args.exp
    
    algo = 'uot_combine_vae'
    statistic_path = f'./saved_info/{args.dataset}/statistic'
    os.makedirs(statistic_path, exist_ok=True)

    
    
    # if args.perturb_percent > 0:
    #     exp_path += f'_{int(args.perturb_percent)}p_{args.perturb_dataset}'
    
    # exp_path += f'/{algo}/{args.phi1}_{args.phi2}/tau_{args.tau}/lrg_{args.lr_g}_lrd_{args.lr_d}/r1_gamma_{args.r1_gamma}/{args.exp}'
    # content_path = f'{exp_path}/content'
    # netG_path = exp_path + "/netG"
    # image_path = exp_path + "/image"
    # loss_file = f"{exp_path}/loss.txt"

    # if rank == 0:
    #     if not os.path.exists(exp_path):
    #         os.makedirs(exp_path)
    #         copy_source(__file__, exp_path)
    #         shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))
    
    #     os.makedirs(content_path, exist_ok=True)
    #     os.makedirs(netG_path, exist_ok=True)
    #     os.makedirs(image_path, exist_ok=True)

    #     # Extract the Python command and its arguments
    #     python_command = ' '.join(sys.argv)
    #     # Extract the CUDA_VISIBLE_DEVICES environment variable
    #     cuda_env_var = os.getenv('CUDA_VISIBLE_DEVICES', '')
    #     # Get the current timestamp
    #     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     # Construct the final command line with time, CUDA_VISIBLE_DEVICES, and the Python command
    #     command_line = f"{timestamp} \nCUDA_VISIBLE_DEVICES={cuda_env_var} python3 {python_command}"
    #     # Write the command line to the output file
    #     output_file = f"{exp_path}/command_history.txt"
    #     with open(output_file, "a") as file:
    #         file.write(command_line + "\n\n\n")
            
    
    # command_line = ' '.join(sys.argv)
    # output_file = f"{exp_path}/command_history.txt"

    # # Include environment variables in the command line
    # env_vars = ' '.join([f"{key}={value}" for key, value in os.environ.items()])
    # command_line_with_env = f"{env_vars} {command_line}"

    # with open(output_file, "a") as file:
    #     file.write(command_line_with_env + "\n")

    

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    for epoch in range(1):
        train_sampler.set_epoch(epoch)
       
        sum_mean_data = torch.zeros([args.num_channels, args.input_size, args.input_size], device=device)
        sum_std_data = torch.zeros([args.num_channels, args.input_size, args.input_size], device=device)
        min_data = torch.full(sum_mean_data.shape, -100, device=device)
        max_data = torch.full(sum_mean_data.shape, 100, device=device)
        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            real_data = vae.encode(real_data).latent_dist.sample().mul_(0.18215).detach()
            sum_mean_data += real_data.mean(dim=0)
        mean_data = sum_mean_data / num_chunks
        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            real_data = vae.encode(real_data).latent_dist.sample().mul_(0.18215).detach()
            real_data = real_data - mean_data
            real_data = real_data ** 2
            sum_std_data += real_data.mean(dim=0)
        std_data = sum_std_data / num_chunks
        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            real_data = vae.encode(real_data).latent_dist.sample().mul_(0.18215).detach()
            min_tensor = torch.min(real_data, dim=0).values
            min_data = torch.min(min_data, min_tensor)
        for iteration, (x, y) in enumerate(tqdm(data_loader)):
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            real_data = vae.encode(real_data).latent_dist.sample().mul_(0.18215).detach()
            max_tensor = torch.max(real_data, dim=0).values
            max_data = torch.max(max_data, max_tensor)

        mean = mean_data.cpu().numpy()
        std = std_data.cpu().numpy()
        min = min_data.cpu().numpy()
        max = max_data.cpu().numpy()
        np.savez(f'{statistic_path}/statistic.npz', mean=mean, std=std, min=min, max=max)
        
        # Load the .npz file
        data = np.load(f'{statistic_path}/statistic.npz')
        # Access the arrays using the custom names
        mean_data = data['mean']
        std_data = data['std']
        min_data = data['min']
        max_data = data['max']
        mean_data = torch.from_numpy(mean_data).to(device)
        std_data = torch.from_numpy(std_data).to(device)
        min_data = torch.from_numpy(min_data).to(device)
        max_data = torch.from_numpy(max_data).to(device)
        # import ipdb; ipdb.set_trace()
        
        
            


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()    
#%%
if __name__ == '__main__':
    args = args_parser()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:        
        init_processes(0, size, train, args)
   
                
