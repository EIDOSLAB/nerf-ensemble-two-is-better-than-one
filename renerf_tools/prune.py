import copy
import torch
import torch.nn.utils.prune as prune

with torch.no_grad():
    def remove_0(model, target_layers, gamma):
            
        '''
            remove parameters
            returns a mask dict
        '''
        
        deltas = []
        for layer in target_layers:
            deltas.append(getattr(model, layer))

        parameters_to_prune = []
        importance_maps = {}
        for layer in target_layers:
            parameters_to_prune.append((model, layer))
            grid = getattr(model, layer)
            taylor = torch.abs(grid.grad * grid)
            importance_maps[layer] = taylor / torch.max(taylor)


        prune.global_unstructured(
            parameters=parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=gamma,
            importance_scores=importance_maps
        )
        
        # build mask dictionary
        mask_keys = [key for key in model.state_dict().keys() if '_mask' in key]
        mask_dict = dict()
        for key in mask_keys:
            mask_dict[key] = model.state_dict()[key]
            

        return mask_dict, deltas
    

@torch.no_grad()
def remove_ensemble(ensemble, target_layers, gamma):
        
    '''
        remove parameters
        returns a mask dict
    '''
    mask_dicts = []
    for model in ensemble.models:
        parameters_to_prune = []
        importance_maps = {}
        for name, module in model.named_modules():
            if name in target_layers:
                parameters_to_prune.append((module, "grid"))
                taylor = torch.abs(module.grid.grad * module.grid)
                importance_maps[name] = taylor / torch.max(taylor)

        prune.global_unstructured(
            parameters=parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=gamma,
            importance_scores=importance_maps
        )
            
            # build mask dictionary
        mask_keys = [key for key in model.state_dict().keys() if '_mask' in key]
        mask_dict = dict()
        for key in mask_keys:
            new_key = key.split('_')[0]
            mask_dict[new_key] = model.state_dict()[key]
        
        mask_dicts.append(mask_dict)
        
    return mask_dicts
    
    
    
with torch.no_grad():
    def remove(model, target_layers, gamma):
        
        '''
            remove parameters
            returns a mask dict
        '''

        
        parameters_to_prune = []
        importance_maps = {}
        for name, module in model.named_modules():
            if name in target_layers:
                parameters_to_prune.append((module, "grid"))
                taylor = torch.abs(module.grid.grad * module.grid)
                importance_maps[name] = taylor / torch.max(taylor)

        prune.global_unstructured(
            parameters=parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=gamma,
            importance_scores=importance_maps
        )
        
        # build mask dictionary
        mask_keys = [key for key in model.state_dict().keys() if '_mask' in key]
        mask_dict = dict()
        for key in mask_keys:
            mask_dict[key] = model.state_dict()[key]
        
        return mask_dict


with torch.no_grad():
    def reinclude_no_prior(model, target_layers, delta, mask_dict, device='cuda'):
        results = dict()
        
        for mask_key in mask_dict:
            strings = mask_key.split('_') # density.grid, mask
            string_main = strings[0] # density.grid
        
            ori_key = string_main + "_orig" # density.grid_orig

            mask = mask_dict[mask_key]
            param_ori = None # contains original grid 

            for param_name, param in model.named_parameters(): 
                if param_name == ori_key:
                    param_ori = param
                    break
            
            abs_imp = torch.abs(param_ori.grad)
            avg_imp = torch.sum(abs_imp * mask) / torch.sum(mask)
            to_include = (inverse_mask(mask) * abs_imp) > avg_imp
            
            results[mask_key] = torch.logical_or(mask, to_include)
    
        return results


with torch.no_grad():
    def reinclude(model, target_layers, delta, mask_dict, device='cuda'):
        '''
            reinclude some parameters
            returns a final mask dict
        '''

        results = dict()
        has_neighbours_3d = torch.nn.AvgPool3d(kernel_size=3, stride=1, padding=1)

        for mask_key in mask_dict:
            
            strings = mask_key.split('_') # density.grid, mask
            string_main = strings[0] # density.grid
        
            ori_key = string_main + "_orig" # density.grid_orig

            mask = mask_dict[mask_key]
            param_ori = None # contains original grid 

            for param_name, param in model.named_parameters(): 
                if param_name == ori_key:
                    param_ori = param
                    break
            
            abs_imp = torch.abs(param_ori.grad)
            avg_imp = torch.sum(abs_imp * mask) / torch.sum(mask)
            
            adding = True
            while adding:
                output = has_neighbours_3d(mask) > 0
                add_neighbours = ((output * inverse_mask(mask) * abs_imp)) > avg_imp
                new_mask = mask.logical_or(add_neighbours)
                num_added = new_mask.sum() - mask.sum()

                if num_added > 0:
                    print(num_added, "added")
                    mask = new_mask.bool().to(torch.float32).clone()
                    mask = mask.to(device)
                else:
                    adding = False

            results[mask_key] = mask
        
        return results

    
def inverse_mask(mask): 
    res = mask.clone()
    res[mask==0] = 1 
    res[mask!=0] = 0
    return res 


