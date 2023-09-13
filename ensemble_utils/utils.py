import copy
import subprocess
import torch
import os
from lib import dvgo, dvgo_ensemble, utils
from lib.dvgo_ensemble import DirectVoxGoEnsemble


def save_models(ensemble, masks, basedir, exp_name, m_names):

    os.makedirs(os.path.join(basedir, exp_name), exist_ok=True)
    
    outpath_list = []
    for model, mask, name in zip(ensemble.models, masks, m_names):

        outpath = os.path.join(basedir, exp_name, f'{name}.tar')

        torch.save({
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'global_step': 0,
            'pruning_mask': mask
        }, outpath)

        print("model saved at ", outpath)
        outpath_list.append(outpath)


    return outpath_list


def quantize(state_dict_path_list, m_names):

    out_list = []
    for i, path in enumerate(state_dict_path_list):
        st_dict = torch.load(path, map_location="cpu")['model_state_dict']
        quantizing_st_dict = {}
        for key in st_dict:
            if not "grid" in key:
                quantizing_st_dict[key] = st_dict[key]
            else:
                non_zeros = (st_dict[key] != 0)*1.0
                quantizing_st_dict[key] = torch.quantize_per_tensor(st_dict[key], 1, 0, torch.qint8)
                st_dict[key] = ((torch.dequantize(quantizing_st_dict[key])) * non_zeros)
        basedir = os.path.split(path)[0]
        outfname = f'{m_names[i]}_quantized_state_dict.tar'
        outpath = os.path.join(basedir, outfname)
        torch.save(st_dict, outpath)
        print(f'Quantizing: saved state_dict at', outpath)
        out_list.append(outpath)

    return out_list


def compress(path):
    subprocess.call(['7z', 'a', str(path) + ".7z", path])

        
def build_ensemble(args, basedir, exp_name, m_names, device):
    models = []
    mask_list = []
    ckpts = []
    dir = os.path.join(basedir, exp_name)
    reloaded = False
    if os.path.exists(dir) and len (os.listdir(dir)) > 0 and not args.no_reload:
        print(f'Detected ckpts in {dir}... reloading models')
        reloaded = True
        for name in m_names:
            ckpt = os.path.join(dir, f'{name}.tar')
            print(ckpt)
            ckpts.append(ckpt)
    else:
        ckpts = args.ckpts
    
    for ckpt in ckpts:
        model, masks = load_ckpt(ckpt, device, args.renerf)
        models.append(model)
        mask_list.append(masks)
        
    return dvgo_ensemble.DirectVoxGoEnsemble(models), mask_list, reloaded


def load_ckpt(ckpt, device, renerf=False, coarse=False):
    if renerf:
        return load_renerf_model(ckpt, device)
    return load_existed_model(ckpt, device, coarse)


def load_renerf_model(ckpt_path, device):
    model_class = dvgo.DirectVoxGO
    model, pruning_mask = utils.load_model_and_pruning_mask(model_class, ckpt_path)
    
    mask_dict = copy.deepcopy(pruning_mask)
    for key in pruning_mask:
        new_key = key.split('_')[0]
        mask_dict[new_key] = mask_dict.pop(key).to(device)
    
    model = model.to(device)
    return model, mask_dict


def load_existed_model(ckpt_path, device, coarse=True):

    model_class = dvgo.DirectVoxGO
    # at the moment only dvgo is supported
    model = utils.load_model(model_class, ckpt_path, coarse)
    model = model.to(device)
    return model, {}


