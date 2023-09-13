import torch
import numpy as np
import random, os, subprocess
import argparse, mmcv
from ensemble_utils.forward import train_ensemble
from ensemble_utils.utils import build_ensemble, compress, load_ckpt, quantize, save_models
from ensemble_utils.evaluation import evaluate_ensemble
from run_dvgo import load_everything, seed_everything
from lib import dvgo, dvgo_ensemble
from lib.dvgo_ensemble import DirectVoxGoEnsemble

import wandb


def config_parser():

    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config")

    parser.add_argument('--ckpts', '--names-list', nargs='+', default=[], 
                        help='paths to models for the ensemble')
    
    parser.add_argument('--renerf', action='store_true', default=None,
                        help='use compressed models')

    parser.add_argument('--no_coarse', action='store_true')

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
    return parser


if __name__ == '__main__':
  
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    basedir = os.path.join('nerf-ensemble-two-is-better-than-one', 'ensemble')
    
    exp_name = '' 
    m_names = []
    for i, ckpt in enumerate(sorted(args.ckpts)):
        m_name = ckpt.split('/')[-2]
        m_names.append(m_name)
        exp_name += m_name
        if i < len(args.ckpts) - 1:
            exp_name += '+'

    seed_everything()

    ensemble, mask_list, reloaded = build_ensemble(args, basedir, exp_name, m_names, device)
    
    ''' load all data (train + test + validation )'''
    cfg.data.white_bkgd = 0
    data_dict = load_everything(cfg)
    
    mean = False
    
    if not args.render_only:
        train_ensemble(ensemble, cfg, cfg.fine_model_and_render, cfg.fine_train,
                        data_dict, 'ensemble-training', mask_list, 0, iters=20000, mean=mean)
    
        state_dict_path_list = save_models(ensemble, mask_list, basedir, exp_name, m_names)
        
        if args.renerf:
            ''' quantize models '''
            outpath_list = quantize(state_dict_path_list, m_names)
                    
            ''' compress models '''
            for path in outpath_list:
                compress(path)
                    
       
    if args.render_test:
        savedir = None
        if args.dump_images:
            savedir = os.path.join(basedir, exp_name, 'render_test')
        
        if args.renerf:
            ''' before quantization'''
            #evaluate_ensemble(ensemble, args, cfg, data_dict, mean=mean)
            
            ''' load quantized dicts '''
            for m, n in zip(ensemble.models, m_names):
                m.load_state_dict(torch.load(os.path.join(basedir, exp_name, f'{n}_quantized_state_dict.tar')))
                
        evaluate_ensemble(ensemble, args, cfg, data_dict, savedir, mean)
