import torch
import os


def quantize_deltas(lr, stage, logfolder):
    quanzited_deltas = []
    deltas = [lr.density_delta, lr.k0_delta]
    for delta in deltas:
        d = delta.cpu()
        non_zeros = (d != 0) * 1.0
        q_d = torch.quantize_per_tensor(d, 1, 0, torch.qint8)
        q_d = ((torch.dequantize(q_d)) * non_zeros)
        quanzited_deltas.append(q_d)
        
    outfname = stage + "_quantized_deltas.tar" 
    outpath = os.path.join(logfolder, outfname)   
    
    torch.save({
        'density_delta': quanzited_deltas[0],
        'k0_delta': quanzited_deltas[1]
        }, outpath)
    
    return outpath
    

def quantizing(model_state_dict, stage, basedir): 
   
    fname = stage + "_state_dict.tar"
    state_dict_path = os.path.join(basedir, fname)
    torch.save(model_state_dict, state_dict_path)

    st_dict = torch.load(state_dict_path, map_location="cpu")
    quantizing_st_dict = {}
    for key in st_dict:
        if not "grid" in key:
            quantizing_st_dict[key] = st_dict[key]
        else:
            non_zeros = (st_dict[key] != 0)*1.0
            quantizing_st_dict[key] = torch.quantize_per_tensor(st_dict[key], 1, 0, torch.qint8)
            st_dict[key] = ((torch.dequantize(quantizing_st_dict[key])) * non_zeros)

    outfname = stage + "_quantized_state_dict.tar"
    outpath = os.path.join(basedir, outfname)
    torch.save(st_dict,outpath)
    print(f'Quantizing ({stage}): saved state_dict at', outpath)

    return outpath