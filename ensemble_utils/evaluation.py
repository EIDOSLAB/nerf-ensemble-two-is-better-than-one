import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from lib import utils, dvgo
import torch.nn.utils.prune as prune
import numpy as np
import imageio
import os
import matplotlib
import matplotlib.pyplot as plt


def evaluate_ensemble(ensemble, args, cfg, data_dict, savedir=None, mean=False):

    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'models': ensemble.models,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
            },
        'mean': mean
        }

    return render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                cfg=cfg,
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=savedir, dump_images=False,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)


@torch.no_grad()
def render_viewpoints(models, render_poses, HW, Ks, ndc, cfg, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, mean=False):

    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs_list = [ [] for i in range(0, len(models)) ]
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    
    mean_psnr, mean_ssim, mean_lpips_alex, mean_lpips_vgg = [-1, -1, -1, -1]

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        rgb_list = []

        for j in range(0, len(models)):

            render_result_chunks = [
                {k: v for k, v in models[j](ro, rd, vd, **render_kwargs).items() if k in keys}
                for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
            ]
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
                for k in render_result_chunks[0].keys()
            }
            rgb = render_result['rgb_marched'].cpu().numpy()
            rgb_list.append(rgb)


        if i==0:
            print('Testing', rgb.shape)


        output = rgb_list[0]
        rgbs_list[0].append(rgb_list[0])
        for j in range(1, len(rgb_list)):
            output += rgb_list[j]
            rgbs_list[j].append(rgb_list[j])

        if mean:
            output = output / len(rgb_list)


        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(output - gt_imgs[i])))
            rgb_list.clear()

            psnrs.append(p)
            if eval_ssim:
                ss = utils.rgb_ssim(output, gt_imgs[i], max_val=1)
                ssims.append(ss)
            if eval_lpips_alex:
                lp = utils.rgb_lpips(output, gt_imgs[i], net_name='alex', device=c2w.device)
                lpips_alex.append(lp)
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))


    if len(psnrs):
        mean = np.mean(psnrs)
        print('Testing psnr', mean, '(avg)')
        mean_psnr = np.mean(psnrs)
        
        if eval_ssim: 
            mean_ssim = np.mean(ssims)
            print('Testing ssim', mean_ssim, '(avg)')
        if eval_lpips_vgg: 
            mean_lpips_vgg = np.mean(lpips_vgg)
            print('Testing lpips (vgg)', mean_lpips_vgg, '(avg)')
        if eval_lpips_alex: 
            mean_lpips_alex = np.mean(lpips_alex)
            print('Testing lpips (alex)', mean_lpips_alex, '(avg)')

    
    if savedir is not None:
        rgb8 = utils.to8b(output)
        filename = os.path.join(savedir, '{:03d}.png'.format(i))
        imageio.imwrite(filename, rgb8)

    return mean_psnr, mean_ssim, mean_lpips_alex, mean_lpips_vgg

