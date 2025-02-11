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


from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

from util.data_process import getCleanData, getMixedData
from util.args_parser import args_parser
from util.utility import copy_source, broadcast_params, q_sample_pairs, sample_posterior, sample_from_model, select_phi
from util.diffusion_coefficients import get_time_schedule, get_sigma_schedule

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
    from score_sde.models.potential import Discriminator_small, Discriminator_large
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
    
    
    train_sampler_duplicate = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)

    data_loader_duplicate = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler_duplicate,
                                               drop_last = True)
    
    netG = NCSNpp(args).to(device)
    

    if args.dataset in ['cifar10', 'mnist', 'stackmnist', 'stl10', 'celeba_64']:
        print("using small discriminator")   
        netD = Discriminator_small(nc = args.num_channels, ngf = args.ngf,
                               t_emb_dim = args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)
    else:
        print("using large discriminator")
        netD = Discriminator_large(nc = args.num_channels, ngf = args.ngf, 
                                   t_emb_dim = args.t_emb_dim,
                                   act=nn.LeakyReLU(0.2)).to(device)
    
    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())
    
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas = (args.beta1, args.beta2))
    
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas = (args.beta1, args.beta2))
    
    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)
    
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.schedule, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.schedule, eta_min=1e-5)
    
    
    
    #ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu])
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu])

    
    exp = args.exp
    
    algo = 'uot_xt_independence_detach_r2'
    exp_path = f'./saved_info/{args.dataset}'

    
    
    if args.perturb_percent > 0:
        exp_path += f'_{int(args.perturb_percent)}p_{args.perturb_dataset}'
    
    exp_path += f'/{algo}/{args.phi1}_{args.phi2}/tau_{args.tau}/bs{batch_size * args.num_process_per_node}/lrg_{args.lr_g}_lrd_{args.lr_d}/r1_gamma_{args.r1_gamma}/{args.exp}'
    netG_path = exp_path + "/netG"
    image_path = exp_path + "/image"
    loss_file = f"{exp_path}/loss.txt"

    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))
    
        os.makedirs(netG_path, exist_ok=True)
        os.makedirs(image_path, exist_ok=True)

        # Extract the Python command and its arguments
        python_command = ' '.join(sys.argv)
        # Extract the CUDA_VISIBLE_DEVICES environment variable
        cuda_env_var = os.getenv('CUDA_VISIBLE_DEVICES', '')
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Construct the final command line with time, CUDA_VISIBLE_DEVICES, and the Python command
        command_line = f"{timestamp} \nCUDA_VISIBLE_DEVICES={cuda_env_var} python3 {python_command}"
        # Write the command line to the output file
        output_file = f"{exp_path}/command_history.txt"
        with open(output_file, "a") as file:
            file.write(command_line + "\n\n\n")
            
    
    # command_line = ' '.join(sys.argv)
    # output_file = f"{exp_path}/command_history.txt"

    # # Include environment variables in the command line
    # env_vars = ' '.join([f"{key}={value}" for key, value in os.environ.items()])
    # command_line_with_env = f"{env_vars} {command_line}"

    # with open(output_file, "a") as file:
    #     file.write(command_line_with_env + "\n")

    
    
    
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    T = get_time_schedule(args, device)
    
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G
        
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0
    
    # get phi star
    phi_star1 = select_phi(args.phi1)
    phi_star2 = select_phi(args.phi2)
    
    total_iterations = len(data_loader)
    tau = args.tau
    upper_d = tau * 2 * args.image_size * args.image_size * args.num_channels
    loss_weight = args.loss_weight
    for epoch in range(init_epoch, args.num_epoch+1):
        train_sampler.set_epoch(epoch)
        train_sampler_duplicate.set_epoch(epoch)
       
        for iteration, ((x, y), (x2, y2)) in enumerate(tqdm(zip(data_loader, data_loader_duplicate), total=total_iterations)):
            for p in netD.parameters():  
                p.requires_grad = True  
        
            
            netD.zero_grad()
            
            #sample from p(x_0)
            real_data = x.to(device, non_blocking=True)
            real_data_2 = x2.to(device, non_blocking=True)
            
            #sample t
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            x_t, x_tp1, _ = q_sample_pairs(coeff, real_data, t)
            _, x_tp1_duplicate, _ = q_sample_pairs(coeff, real_data_2, t)
            
    
            # train with real
            x_t.requires_grad = True
            D_real = netD(x_t, t)
            errD_real = phi_star1(D_real - upper_d)
            errD_real = errD_real.mean()
            errD_real.backward(retain_graph=True)

            # train with fake
            latent_z = torch.randn(batch_size, nz, device=device)
            x_0_predict = netG(x_tp1.detach(), t, latent_z).detach()
            x_pos_sample, _ = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            x_pos_sample.requires_grad = True
            D_fake = netD(x_pos_sample, t)
            errD_fake = phi_star2(-D_fake)
            errD_fake = errD_fake.mean()
            errD_fake.backward(retain_graph=True)

            errD = D_fake - tau * torch.sum(((x_pos_sample-x_tp1_duplicate.detach()).view(x_tp1_duplicate.detach().size(0), -1))**2, dim=1) - D_real
            errD = (loss_weight * errD).mean()
            errD.backward(retain_graph=True)
            
            
            if global_step % args.lazy_reg == 0:

                grad_real = torch.autograd.grad(
                                outputs=D_real.sum(), inputs=x_t, create_graph=True
                                )[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()

                grad_fake = torch.autograd.grad(
                                outputs=D_fake.sum(), inputs=x_pos_sample, create_graph=True
                                )[0]
                grad_penalty = (grad_fake.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = args.r1_gamma / 2 * grad_penalty
                grad_penalty.backward()
            
                
    
            
            # errD = errD_fake
            # Update D
            optimizerD.step()
            
        
            #update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            
            t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
            
            
            x_t, x_tp1, _ = q_sample_pairs(coeff, real_data, t)
            _, x_tp1_duplicate, _ = q_sample_pairs(coeff, real_data_2, t)
                
            
            latent_z = torch.randn(batch_size, nz,device=device)
            
            
                
           
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_pos_sample, noise = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
            
            D_fake = netD(x_pos_sample, t)
               
            errG = tau * torch.sum(((x_pos_sample-x_tp1_duplicate.detach()).view(x_tp1_duplicate.detach().size(0), -1))**2, dim=1) - D_fake
            errG = errG.mean()
            
            errG.backward()
            optimizerG.step()
                
           
            
            global_step += 1
            if iteration % 100 == 0:
                if rank == 0:
                    loss_content = 'epoch {} iteration {}, G Loss: {}, D Loss: {}'.format(epoch,iteration, errG.item(), errD.item())
                    with open(loss_file, "a") as file:
                        file.write(loss_content + "\n")
                    print(loss_content)
        
        if not args.no_lr_decay:
            
            schedulerG.step()
            schedulerD.step()
        
        if rank == 0:
            if epoch % 10 == 0:
                torchvision.utils.save_image(x_pos_sample, os.path.join(image_path, 'xpos_epoch_{}.png'.format(epoch)), normalize=True)
            
            x_t_1 = torch.randn_like(real_data)
            fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, T, args)
            if epoch % 10 == 0:
                torchvision.utils.save_image(fake_sample, os.path.join(image_path, 'sample_discrete_epoch_{}.png'.format(epoch)), normalize=True)
            
            if args.save_content:
                if epoch % args.save_content_every == 0:
                    print('Saving content.')
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}
                    
                    torch.save(content, os.path.join(exp_path, 'content.pth'))
                
            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
                    
                torch.save(netG.state_dict(), os.path.join(netG_path, 'netG_{}.pth'.format(epoch)))
                if args.use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            


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
   
                
