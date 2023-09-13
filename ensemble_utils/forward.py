import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from lib import utils, dvgo
from lib.masked_adam import MaskedAdam
import torch.nn as nn
import numpy as np


def compute_loss(render_result_list, target, mean=False):
    out = render_result_list[0]['rgb_marched']
    for i in range(1, len(render_result_list)):
        out += render_result_list[i]['rgb_marched']
    if mean:
        out = out / len(render_result_list)
    loss = F.mse_loss(out, target)
    return loss


def create_optimizer(ensemble, cfg_train, global_step):

    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []

    start = 0
    end = len(ensemble.models)
    
    # for each model
    for i in range(start, end):

        for k in cfg_train.keys():
            if not k.startswith('lrate_'):
                continue
            k = k[len('lrate_'):]

            if not hasattr(ensemble.models[i], k):
                continue

            param = getattr(ensemble.models[i], k)
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

    if hasattr(ensemble, 'rgbnet'):
        lr_nn = getattr(cfg_train, f'lrate_rgbnet') * decay_factor
        param_group.append({'params': ensemble.rgbnet.parameters(), 'lr': lr_nn, 'skip_zero_grad': False})


    return MaskedAdam(param_group)


@torch.no_grad()
def apply_masks(dvgo_ensemble, mask_list):

    for i in range(0, len(dvgo_ensemble.models)):
        for name, module in dvgo_ensemble.models[i].named_parameters():
            key = name.split("_")[0]
            if key in mask_list[i].keys():
                module.data.mul_(mask_list[i][key])
                # if key == 'density.grid':
                #     dvgo_ensemble.models[i].mask_cache.mask.mul_(mask_list[i][key][0, 0].bool())


def forward_ensemble(dvgo_ensemble, cfg, cfg_model, cfg_train, data_dict, stage, mask_list, step, iters=20000, mean=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data needed for training
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]


    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 0,
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

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_val],
                    HW=HW[i_val], Ks=Ks[i_val],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=dvgo_ensemble.models[-1], render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_val],
                HW=HW[i_val], Ks=Ks[i_val], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_val],
                HW=HW[i_val], Ks=Ks[i_val], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
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


    retain = True
    
    # all models do the same number of iterations - starts always at 1
    for global_step in trange(1, 1 + iters):

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

        # volume rendering
        render_result_list = dvgo_ensemble(
                rays_o, rays_d, viewdirs,
                global_step=global_step,
                **render_kwargs)

        loss = compute_loss(render_result_list, target, mean=mean)
        
        if global_step == iters: # last iteration, no need to retain graph
            retain = False 
        
        loss.backward(retain_graph=retain)  
       


def train_ensemble(dvgo_ensemble, cfg, cfg_model, cfg_train, data_dict, stage, mask_list,
                   step, iters=20000, mean=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data needed for training
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]


    optimizer = create_optimizer(dvgo_ensemble, cfg_train, step)


    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 0,
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
                    model=dvgo_ensemble.models[-1], render_kwargs=render_kwargs)
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

    if mask_list:
        apply_masks(dvgo_ensemble, mask_list)
        

    # all models do the same number of iterations - starts always at 1
    for global_step in trange(1, 1 + iters):
        
        
        for model in dvgo_ensemble.models:
            if model.mask_cache is not None and (global_step + 500) % 1000 == 0:
                model.update_occupancy_cache()

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

        # volume rendering
        render_result_list = dvgo_ensemble(
                rays_o, rays_d, viewdirs,
                global_step=global_step,
                **render_kwargs)

        loss = compute_loss(render_result_list, target, mean=mean)
        optimizer.zero_grad(set_to_none=True)
        psnr = utils.mse2psnr(loss.detach())


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
        if global_step % 500 == 0:
            tqdm.write(f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / '
                    f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_list):5.2f}')
            psnr_list = []

        if mask_list:
            apply_masks(dvgo_ensemble, mask_list)
