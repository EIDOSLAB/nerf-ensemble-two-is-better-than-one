from renerf_tools import forward
import torch
import os, copy, subprocess
import shutil
import numpy as nppretrained_path
import argparse
import random
import numpy as np
from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data
import imageio
from tqdm import tqdm, trange


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument('--wdb_name', required=True,
                        help='wandb project name')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    # my params
    parser.add_argument("--ckpt")
    # parser.add_argument("--expname", required=True)
    # parser.add_argument("--datadir", required=True)
    # parser.add_argument("--vdim", default=160, type=int)
    # parser.add_argument("--n_iter_fine", default=20000, type=int)
    
    return parser


def seed_everything(seed):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def compress(path):
    subprocess.call(['7z', 'a', str(path) + ".7z", path]) 


def load_everything(cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # remove useless field
    kept_keys = {
            'hwf', 'HW', 'Ks', 'near', 'far', 'near_clip',
            'i_train', 'i_val', 'i_test', 'irregular_shape',
            'poses', 'render_poses', 'images'}
    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    # construct data tensor
    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    
    return data_dict


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs, cfg,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    psnrs = []
    psnr_mean = 0

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
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        rgbs.append(rgb)

        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            

    if len(psnrs):
        psnr_mean = np.mean(psnrs)


    if savedir is not None and dump_images:
        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
    
    return psnr_mean

            
def evaluate_model(model, data_dict, cfg, testsavedir):
    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
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
        'cfg': cfg
    }
    os.makedirs(testsavedir, exist_ok=True)
    print('All results are dumped into', testsavedir)
    psnr = render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=testsavedir,
                **render_viewpoints_kwargs)
    return psnr