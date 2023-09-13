import time, os
import torch
import numpy as np
from lib import dvgo, dbvgo, dcvgo, dmpigo, utils
from tqdm import tqdm, trange
from torch_efficient_distloss import flatten_eff_distloss
import torch.nn.functional as F
from lib.masked_adam import MaskedAdam
from renerf_tools import renerf_utils
import imageio
import torch.nn as nn


def create_optimizer(model, cfg_train, global_step):
    
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                param = param.parameters()
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    
    return MaskedAdam(param_group)


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def forward(model, cfg, cfg_model, cfg_train, data_dict, stage):
    '''
        forward pass
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data needed for training
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]
    
    # init rendering setup (same for every model)
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    
    
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_val]
        else:
            rgb_tr_ori = images[i_val].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        # same rays for all models 
        # if cfg_train.ray_sampler == 'in_maskcache':
        #     rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
        #             rgb_tr_ori=rgb_tr_ori,
        #             train_poses=poses[i_train],
        #             HW=HW[i_train], Ks=Ks[i_train],
        #             ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
        #             flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
        #             model=model, render_kwargs=render_kwargs)
        # elif cfg_train.ray_sampler == 'flatten':
        #     rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
        #         rgb_tr_ori=rgb_tr_ori,
        #         train_poses=poses[i_train],
        #         HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
        #         flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        # else:
        rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
    

    torch.cuda.empty_cache()
    global_step = -1    
    
    iters = 1000
    retain = True


    for global_step in trange(1, 1 + iters):
            
        # random sample rays
        # if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
        #     sel_i = batch_index_sampler()
        #     target = rgb_tr[sel_i]
        #     rays_o = rays_o_tr[sel_i]
        #     rays_d = rays_d_tr[sel_i]
        #     viewdirs = viewdirs_tr[sel_i]
        # elif cfg_train.ray_sampler == 'random':
        sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
        sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
        sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
        target = rgb_tr[sel_b, sel_r, sel_c]
        rays_o = rays_o_tr[sel_b, sel_r, sel_c]
        rays_d = rays_d_tr[sel_b, sel_r, sel_c]
        viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        # else:
        #     raise NotImplementedError

            
        # volume rendering
        render_result = model(
            rays_o, rays_d, viewdirs,
            global_step=global_step, is_train=True,
            **render_kwargs)
                

        loss = F.mse_loss(target, render_result['rgb_marched'])
        
        if global_step == iters: # last iteration, no need to retain graph
            retain = False 
        
        loss.backward(retain_graph=retain)  
        

    return model


def fine_tune(model, args, cfg, cfg_model, cfg_train, data_dict, stage, mask, step):
    '''
        fine-tuning on train dataset
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data needed for training
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]
    
    model.zero_grad()
    
    with torch.no_grad():
        if mask:
            for name, module in model.named_parameters():
                m_name = name.split('_')[0] + '_mask'
                if m_name in mask.keys():
                    module.data.mul_(mask[m_name]) 
                        
            #         if m_name == 'density.grid_mask':
            #             model.mask_cache.mask.mul_(mask[m_name][0, 0].bool())
            # model.density.grid.data.mul(mask['density.grid_mask'])
            # model.k0.grid.data.mul(mask['density.grid_mask'])
            # model.mask_cache.mask.mul_(mask['density.grid_mask'][0, 0].bool())

    
    ##### optimizer creation #####
    optimizer = create_optimizer(model, cfg_train, step) 
    
    # init rendering setup (same for every model)
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }
    
    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=model, render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    # same for all models
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()
    
    # now we are ready for training
    torch.cuda.empty_cache()
    psnr_list = []
    global_step = -1    
    
    # all models do the same number of iterations - starts always at 1
    for global_step in trange(1, 1 + cfg_train.N_iters):
            
        # random sample rays
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError
            
        render_result = model(
                rays_o, rays_d, viewdirs,
                global_step=global_step, is_train=True,
                **render_kwargs)
                
        optimizer.zero_grad(set_to_none=True)
        loss =  F.mse_loss(render_result['rgb_marched'], target)  
        psnr = utils.mse2psnr(loss.detach())     
        
        
        if cfg_train.weight_entropy_last > 0:
            pout = render_result['alphainv_last'].clamp(1e-6, 1-1e-6)
            entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
            loss += cfg_train.weight_entropy_last * entropy_last_loss
        if cfg_train.weight_nearclip > 0:
            near_thres = data_dict['near_clip'] / model.scene_radius[0].item()
            near_mask = (render_result['t'] < near_thres)
            density = render_result['raw_density'][near_mask]
            if len(density):
                nearclip_loss = (density - density.detach()).sum()
                loss += cfg_train.weight_nearclip * nearclip_loss
        if cfg_train.weight_distortion > 0:
            n_max = render_result['n_max']
            s = render_result['s']
            w = render_result['weights']
            ray_id = render_result['ray_id']
            loss_distortion = flatten_eff_distloss(w, s, 1/n_max, ray_id)
            loss += cfg_train.weight_distortion * loss_distortion
        if cfg_train.weight_rgbper > 0:
            rgbper = (render_result['raw_rgb'] - target[render_result['ray_id']]).pow(2).sum(-1)
            rgbper_loss = (rgbper * render_result['weights'].detach()).sum() / len(rays_o)
            loss += cfg_train.weight_rgbper * rgbper_loss 
         
            
        # gradient descent
        loss.backward()
        optimizer.step()
        
        psnr_list.append(psnr.item())
            
        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor
                
        # check log 
        if global_step % args.i_print == 0:
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                    f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_list):5.2f}')
            psnr_list = []
        
        #apply mask
        with torch.no_grad():
            if mask:
                for name, module in model.named_parameters():
                    m_name = name.split('_')[0] + '_mask'
                    if m_name in mask.keys():
                        module.data.mul_(mask[m_name]) 
                            
                #         if m_name == 'density.grid_mask':
                #             model.mask_cache.mask.mul_(mask[m_name][0, 0].bool())
                # model.density.grid.data.mul(mask['density.grid_mask'])
                # model.k0.grid.data.mul(mask['density.grid_mask'])
                # model.mask_cache.mask.mul_(mask['density.grid_mask'][0, 0].bool())

    
    return model, global_step

