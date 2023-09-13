'''

 Re:NeRF original implementation
 
'''

import torch
import mmcv
import os
import copy
from renerf_tools import renerf_utils, forward, prune, quantize
from lib import utils, dvgo
import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb


def load_model(model_class, ckpt_path, reload_coarse=True):
    ckpt = torch.load(ckpt_path)
    if reload_coarse:
        for i in range(len(ckpt_path) - 1, 0, -1):
            if ckpt_path[i] == '/':
                break
        coarse_final_path = ckpt_path[0:i+1] + 'coarse_last.tar'
        ckpt['model_kwargs']['mask_cache_path'] = coarse_final_path
    else:
        ckpt['model_kwargs']['mask_cache_path'] = None
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    return model


def load_existed_model(ckpt_path):
    model_class = dvgo.DirectVoxGO
    model = load_model(model_class, ckpt_path).to(device)
    return model


def save_model(model, logfolder, stage, pruning_mask=None):
            model_state_dict = model.state_dict()   
            keys = list(model_state_dict.keys())
            for key in keys:
                if '_orig' in key:
                    name = key.split('_')[0]
                    model_state_dict[name] = model_state_dict.pop(key, None)
                    
                elif '_mask' in key:
                    model_state_dict.pop(key, None)
                        
            torch.save({
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model_state_dict,
                'global_step': 0,
                'pruning_mask': pruning_mask
                }, f'{logfolder}/{stage}_finalgrid.tar')
            
            print("model saved at ", f'{logfolder}/{stage}_finalgrid.tar')
            return model_state_dict    


def evaluate_model(model, data_dict):
    
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
        }
        
    return render_viewpoints(
                render_poses=data_dict['poses'][data_dict['i_test']],
                HW=data_dict['HW'][data_dict['i_test']],
                Ks=data_dict['Ks'][data_dict['i_test']],
                gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                savedir=None, dump_images=args.dump_images,
                eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                **render_viewpoints_kwargs)


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    
    ''' Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

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
            
        if i==0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            #print(p)
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(rgb, gt_imgs[i], net_name='vgg', device=c2w.device))
            
    if len(psnrs):
        mean = np.mean(psnrs)
        print('Testing psnr', mean, '(avg)')
        if eval_ssim: print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg: print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex: print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')
        return mean
    
    
def reNerf(target_layers, cfg, args):

    data_dict = renerf_utils.load_everything(cfg)

    logfolder, fname = os.path.split(args.ckpt)
    model = None
    pretrained_path = args.ckpt
    best_psnr = 0.0

    delta = 0.5
    prune_per_step = 0.5
    iteration = 1
    re_include = False

    mask_dict = None


    while True:
        
        gamma = round(1.0 - (prune_per_step)**iteration, 5)

        print("pruning with gamma =", gamma, " and delta =", delta)

        stage = 'reNeRF(test)-' + str(iteration) 
        pre_stage = 'reNeRF(test)-' + str(iteration-1)
 
        if iteration > 1:
            pretrained_path = f'{logfolder}/{pre_stage}_finalgrid.tar'
        
        
        # pretrained model loading      
        model = load_existed_model(pretrained_path)

        #baseline (compression without pruning)
        if iteration == 1:
            # model_state_dict = save_model(model, logfolder, 'baseline')
            # outpath = quantize.quantizing(model_state_dict, 'baseline', logfolder)
            # renerf_utils.compress(outpath)
            # size = os.path.getsize(outpath + '.7z')/1000000
            # print("baseline (MB):", size)
            best_psnr = evaluate_model(model, data_dict)
            
        
        m_copy = copy.deepcopy(model)
        model = forward.forward(m_copy, cfg, cfg.fine_model_and_render, cfg.fine_train, data_dict, 'prune')

        # re-move
        mask_dict = prune.remove(m_copy, target_layers, gamma)
        
        if re_include:
            mask_dict = prune.reinclude(m_copy, target_layers, delta, mask_dict)
           
        torch.cuda.empty_cache()

        model, m_step = forward.fine_tune(model, 
                  args, 
                  cfg, 
                  cfg.fine_model_and_render, 
                  cfg.fine_train, 
                  data_dict, 
                  'fine-tune', 
                  mask_dict,
                  0)      
        
        #step += m_step     
        # saving new state_dict
        model_state_dict = save_model(model, logfolder, stage, mask_dict)
        
        # quantization and compression of state_dict
        outpath = quantize.quantizing(model_state_dict, stage, logfolder)
        renerf_utils.compress(outpath)
        
        size = os.path.getsize(outpath + '.7z')/1000000
        
        # reload (quantized dict) model  before testing
        model = load_existed_model(f'{logfolder}/{stage}_finalgrid.tar')
        model.load_state_dict(torch.load(outpath), strict=False)
        
        torch.cuda.empty_cache()
        
        # evaluation
        psnr = evaluate_model(model, data_dict)
        print("Current psnr =", psnr)


        if psnr < best_psnr - 2.0:
            print("End pruning. Prune percentage =", gamma, ".",
                     "Current PSNR =", psnr, " Best PSNR =" , best_psnr)
            break
        
        iteration += 1
        

     
if __name__ == '__main__':
    

    parser = renerf_utils.config_parser()
    args = parser.parse_args()    
    cfg = mmcv.Config.fromfile(args.config)
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    renerf_utils.seed_everything(args.seed)
    
    target_layers = ['density', 'k0']

    reNerf(target_layers, cfg, args)