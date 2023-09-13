'''
    Re:NeRF implementation adapted to jointly prune an ensemble
'''

import torch
import mmcv
import os, random
import copy
from ensemble_utils.evaluation import evaluate_ensemble
from ensemble_utils.forward import forward_ensemble, train_ensemble
from ensemble_utils.utils import build_ensemble, compress, save_models, quantize
from main import config_parser
from renerf_tools import renerf_utils, forward, prune
from lib import utils, dvgo, dcvgo, dmpigo
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import wandb

from run_dvgo import load_everything, seed_everything

                
def renerf_ensemble(ensemble, target_layers, m_names):
    
    logfolder = os.path.join(basedir, exp_name)
    # os.makedirs(f'{logfolder}/renerf', exist_ok=True)
    best_psnr = 0.0
    delta = 0.99
    prune_per_step = 0.5
    iteration = 1
    step = 0
    
    pretrained_paths = m_names
    while True:
            
        gamma = round(1.0 - (prune_per_step)**iteration, 5)
        gamma = 0.95

        print("pruning with gamma =", gamma, " and delta =", delta)

        stage = 'reNeRF-' + str(iteration) 
        pre_stage = 'reNeRF-' + str(iteration-1) 
        

        if iteration > 1:
            pretrained_paths = [f'renerf/{m_name}' for m_name in filenames]
            
        args.no_reload=False
        ensemble, mask_list, reloaded = build_ensemble(args, basedir, exp_name, pretrained_paths, device)
        
        if iteration == 1:
        #     args.eval_lpips_alex = False
        #     args.eval_lpips_vgg = False
        #     args.eval_ssim = True
            best_psnr, ssim, _, _ = evaluate_ensemble(ensemble, args, cfg, data_dict, None, mean)
        #     wandb.log({'psnr': best_psnr, 'ssim': ssim })
            
        
        ens_copy = copy.deepcopy(ensemble)
        forward_ensemble(ens_copy, cfg, cfg.fine_model_and_render, cfg.fine_train, data_dict, stage, mask_list, step, iters=1000, mean=mean)

        # remove
        mask_list = prune.remove_ensemble(ens_copy, target_layers, gamma)

        del(ens_copy)
        torch.cuda.empty_cache()

        train_ensemble(ensemble, cfg, cfg.fine_model_and_render, cfg.fine_train,
                    data_dict, 'ensemble-training', mask_list, 0, iters=2000, mean=mean)

        filenames = [m_name + '-' + stage for m_name in m_names]
        state_dict_path_list = save_models(ensemble, mask_list, logfolder, 'renerf', filenames)

        ''' quantize models '''
        outpath_list = quantize(state_dict_path_list, filenames)
                    
        ''' compress models '''
        for path in outpath_list:
            compress(path)
        
        ''' reload quantized models '''
        for m, path in zip(ensemble.models, outpath_list):
                m.load_state_dict(torch.load(path))
        
        args.eval_lpips_alex = True
        args.eval_lpips_vgg = True
        args.eval_ssim = True        
        
        psnr, ssim, lpips_alex, _ = evaluate_ensemble(ensemble, args, cfg, data_dict, None, mean)
        
        sizes = [os.path.getsize(f'{path}.7z')/1024000 for path in outpath_list]
        
        #wandb.log({'gamma': gamma, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips_alex, 'size_0': sizes[0], 'size_1': sizes[1]})
        
        if psnr < best_psnr - 2.0:
            print("End pruning. Prune percentage =", gamma, ".",
                    "Current PSNR =", psnr, " Best PSNR =" , best_psnr)
            break
        
        iteration += 1
        
     
if __name__ == '__main__':
    

    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
    basedir = '/scratch/nerf/NeRF-Ensemble/ensemble'
    
    exp_name = '' # lego_160+lego_180
    m_names = [] # [lego_160, lego_180]
    
    objs = ['lego']
    res_pair = [[160, 256]]
    mean = False
    paths = []
    for obj in objs:
        for pair in res_pair:
            for p in pair:
                paths.append(f'/scratch/nerf/NeRF-Ensemble/ckpts/nerf/{obj}/{obj}_{p}/fine_last.tar')
            
   
            args.ckpts = paths
                
            for i, ckpt in enumerate(sorted(paths)):
                m_name = ckpt.split('/')[-2]
                m_names.append(m_name)
                exp_name += m_name
                if i < len(args.ckpts) - 1:
                    exp_name += '+'

            seed_everything()
            
            exp_name = f'mean={mean}-{exp_name}'

            ensemble, mask_list, reloaded = build_ensemble(args, basedir, exp_name, m_names, device)
            
            wandb.init(project='ablation', name=exp_name)
                
            ''' load all data (train + test + validation )'''
            cfg.data.white_bkgd = 0
            data_dict = load_everything(cfg)
                
            train_ensemble(ensemble, cfg, cfg.fine_model_and_render, cfg.fine_train,
                                data_dict, 'ensemble-training', None, 0, iters=2000, mean=mean)
                
            if pair[0] == pair[1]:
                m_names = [f'{name}_{i}' for i,name in enumerate(m_names)]
            
            state_dict_path_list = save_models(ensemble, mask_list, basedir, exp_name, m_names)
                
            target_layers = ['density', 'k0']
                
            renerf_ensemble(ensemble, target_layers, m_names)
            wandb.finish()
            m_names = []
            exp_name = ''
            paths = []
    