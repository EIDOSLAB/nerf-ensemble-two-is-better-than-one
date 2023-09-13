'''
    This is just a wrapper to DVGO. Use this to easily train multiple model independently.
'''
    

from lib import dvgo, dbvgo, dcvgo, dmpigo, utils
import torch
import numpy as np
import mmcv
import os, sys, copy, glob, json, time, random, argparse
from run_dvgo import config_parser, load_everything, render_viewpoints, seed_everything, train
import wandb


if __name__ == '__main__':
    

    basedir = os.path.join('nerf-ensemble-two-is-better-than-one', 'ckpts')
    dataset_name = 'nerf' # available: blendedmvs, co3d, deepvoxels, lf, llff, nerf, nsvg, tankstemple
    objs = ['lego', 'mic', 'ship', 'drums', 'chair', 'materials', 'hotdog', 'ficus'] # scene of the dataset you want to use
    resolutions = [160, 256] # specify the model resolution (160 means 160x160x160)
    parser = config_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
            
    seed_everything(777)

    for o in objs:
        args.config = os.path.join(basedir, 'configs', dataset_name, f'{o}.py')
        cfg = mmcv.Config.fromfile(args.config)
        for r in resolutions:
            
            cfg.expname = f'{o}_{r}'
            cfg.basedir = os.path.join(basedir, 'ckpts', dataset_name, o)
            cfg.fine_model_and_render.num_voxels = r**3
            cfg.fine_model_and_render.num_voxels_base = r**3
            
            data_dict = load_everything(cfg)
            train(args, cfg, data_dict)
            
            if args.render_test or args.render_train or args.render_video:
                if args.ft_path:
                    ckpt_path = args.ft_path
                else:
                    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
                ckpt_name = ckpt_path.split('/')[-1][:-4]
                if cfg.data.ndc:
                    model_class = dmpigo.DirectMPIGO
                elif cfg.data.unbounded_inward:
                    model_class = dcvgo.DirectContractedVoxGO
                else:
                    model_class = dvgo.DirectVoxGO
                model = utils.load_model(model_class, ckpt_path).to(device)
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
                
                if args.render_test:
                    testsavedir = os.path.join(cfg.basedir, cfg.expname, 'render_test')
                    os.makedirs(testsavedir, exist_ok=True)
                    print('All results are dumped into', testsavedir)
                    mean_psnr, mean_ssim, mean_lpips_vgg = render_viewpoints(
                            render_poses=data_dict['poses'][data_dict['i_test']],
                            HW=data_dict['HW'][data_dict['i_test']],
                            Ks=data_dict['Ks'][data_dict['i_test']],
                            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                            savedir=testsavedir, dump_images=args.dump_images,
                            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                            **render_viewpoints_kwargs)

            
            