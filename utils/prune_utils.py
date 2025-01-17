import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import re
import os
import copy
from torch.ao.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class PruneUtils:

    def __init__(self):
       pass

    #################################
    # Layer Wise Sparsity Functions #
    #################################

    def apply_sparsity_module(self, module, prune_type, sparsity_level):
        if prune_type == 'random':
            prune.random_unstructured(module, name='weight', amount=sparsity_level)
        elif prune_type == 'l1':
            prune.l1_unstructured(module, name='weight', amount=sparsity_level)
        elif prune_type == 'l1_structured':
            prune.ln_structured(module, name="weight", amount=sparsity_level, n=1, dim=0)
        elif prune_type == 'l2_structured':
            prune.ln_structured(module, name="weight", amount=sparsity_level, n=2, dim=0)
        else:
            raise Exception(f"Unsupported Pruning Type {prune_type}")

    def apply_sparsity_layer_wise(self, model, 
                                        sparsity_level, 
                                        prune_type,
                                        module_types=[nn.Linear, nn.Conv2d],
                                        permanent_prune_remove=False):
        module_prune_list = []
        for name, module in model.named_modules():
            for module_type in module_types:
                if isinstance(module, module_type):
                    self.apply_sparsity_module(module=module, prune_type=prune_type, sparsity_level=sparsity_level)
                    module_prune_list.append([module, 'weight'])

        if permanent_prune_remove:
            self._apply_prune_remove(module_prune_list)
            
        return model, module_prune_list

    ##########################
    # Global pruning methods #
    ##########################

    def _get_module_layers_for_global_pruning(self, model, 
                                                    module_types=[nn.Linear, nn.Conv2d], 
                                                    prefix_name='model'):
        """
        Utility Function
        Prints 'Referenceable' names of all modules (in module types)
        """
        module_list = []
        for name, module in model.named_modules():
            for module_type in module_types:
                if isinstance(module, module_type):
                    module_name = f"{prefix_name}.{name}"
                    module_name_split = module_name.split('.')
                    module_name_refined = ''
                    for i in range(0, len(module_name_split)):

                        if module_name_split[i].isnumeric():
                            module_name_refined += '[' + module_name_split[i] + ']'
                        else:
                            module_name_refined += module_name_split[i]

                        if i + 1 < len(module_name_split) and not module_name_split[i+1].isnumeric():
                            module_name_refined += '.'
                    module_list.append([eval(module_name_refined), 'weight'])
        return module_list


    def apply_sparsity_global(self, model, 
                                    sparsity_level, 
                                    prune_type,
                                    parameters_to_prune=None,
                                    module_types=[nn.Linear, nn.Conv2d],
                                    permanent_prune_remove=False):
        prune_type_dict = {'random_unstructred': prune.RandomUnstructured, 
                           'l1_unstructred': prune.L1Unstructured,
                          }
        if prune_type not in prune_type_dict.keys():
            raise Exception(f"Unsupported Prune Type for Global {prune_type}")

        if parameters_to_prune is None:
            parameters_to_prune = self._get_module_layers_for_global_pruning(model=model, 
                                                                             module_types=module_types, 
                                                                             prefix_name='model')
        prune.global_unstructured(parameters_to_prune,
                                  pruning_method=prune_type_dict[prune_type],
                                  amount=sparsity_level)

        if permanent_prune_remove:
            self._apply_prune_remove(parameters_to_prune)

        return model, parameters_to_prune

    #################
    # General Tools #
    #################

    def _apply_prune_remove(self, parameters_to_prune):
        for p in parameters_to_prune:
            # p takes the form (module, 'weight')
            prune.remove(*p)


    def calculate_model_size(self, model):
        sd = model.state_dict()

        for item in sd:
            sd[item] = model.state_dict()[item].to_sparse()

        torch.save(sd, "pruned_model.pt")
        print(f'{os.path.getsize("pruned_model.pt")/1e6} MB')

        size_mb = os.path.getsize("pruned_model.pt")/1e6

        os.remove("pruned_model.pt")
        return size_mb

    def _rewind_load_prev_weights(self, model, rewind_ckpt):
                
        init_ckpt = torch.load(rewind_ckpt)
        
        # Pre-Trained (Rewind) model weights
        init_weights = init_ckpt['model_state_dict']
        
        # Current model weights (with weightes_orig)
        model_sd_copy = copy.deepcopy(model.state_dict())

        model_sd_copy_keys = model_sd_copy.keys()
        ref_orig_weights_to_replace = []

        for key in model_sd_copy_keys:
            if 'weight_orig' not in key:
                continue
            key_split = key.split('.')[:-1]
            key_mod = '.'.join(key_split)

            ref_orig_weights_to_replace.append(key_mod)

        init_updated = {}
        for k, v in init_weights.items():

            k_split = k.split('.')[:-1]
            k_mod = '.'.join(k_split)
            
            if k_mod not in ref_orig_weights_to_replace or 'weight' not in k:
                init_updated[k] = v
                continue

            k_new = k + '_orig'
            init_updated[k_new] = v

        # Update Model Weights
        model_sd_copy.update(init_updated)
        model.load_state_dict(model_sd_copy)

        return model

    def sparse_apply(self, model):
        sd = model.state_dict()
        # print(sd)
        for item in sd:
            # print("item", item[0])
                # if 'weight' in item: # shortcut, this assumes you pruned (and removed reparameterization for) all weight parameters
            #     print("sparsifying", item)
            sd[item] = model.state_dict()[item].to_sparse()
        return model
        

class SparsityCalculator:

    def __init__(self):
        pass

    @staticmethod
    def process_name(name, model_var_name):

        search_regexp = re.compile('\.\d\.')  
        iterator = search_regexp.finditer(name)
        name_rename = list(name)
        for match in iterator:
            st, end = match.span()
            name_rename[int(st)] = '['
            name_rename[int(end)-1] = '].'
        name_rename = ''.join(name_rename)

        name_rename = name_rename.replace('_orig', '')

        return f"{model_var_name}.{name_rename}"
        
    @staticmethod
    def calculate_sparsity_pruned_model(model):

        layer_wise_sparsity = {}

        num_sparse_elem = 0
        num_elem = 0

        tot_elem_sparse = 0

        for name, weight in model.named_parameters():
            name_processed = SparsityCalculator.process_name(name, "model")
            tens_weight = eval(name_processed)

            curr_n_elem = tens_weight.nelement()
            curr_sparse_elem = torch.sum(tens_weight == 0).item()

            layer_wise_sparsity[name_processed] = (curr_sparse_elem * 100.0)/curr_n_elem

            num_sparse_elem += curr_sparse_elem
            num_elem += curr_n_elem

            if curr_sparse_elem > 0:
                tot_elem_sparse += curr_n_elem

        tot_sparsity = num_sparse_elem / num_elem
        tot_sparsity_pruned_layers = num_sparse_elem/ tot_elem_sparse

        return layer_wise_sparsity, tot_sparsity, tot_sparsity_pruned_layers


class QuantizationUtils:

    def __init__(self):
        pass

    @staticmethod
    def dynamic_quantization(model, layers={nn.Linear, nn.Conv2d}):
        model_quantized = torch.quantization.quantize_dynamic(
            model, layers, dtype=torch.qint8
            )
        return model_quantized

    @staticmethod
    def calibrate(model, data_loader):
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs = data["image"].to('cpu')
                labels = data["heatmaps"].to('cpu')
                model(inputs)
        return model

    @staticmethod
    def static_quantize_model(model, data_loader):
        model=model.to('cpu')
        model.eval()
        qconfig = get_default_qconfig('fbgemm')
        qconfig_dict = {"": qconfig}
        prepared_model = prepare_fx(model, qconfig_dict)
        quantized_model = QuantizationUtils.calibrate(prepared_model, data_loader)
        quantized_model = convert_fx(quantized_model)
        return quantized_model

    @staticmethod
    def get_size_of_model(model, label=""):
        torch.save(model.state_dict(), "temp.p")
        size=os.path.getsize("temp.p")
        print("model: ",label,' \t','Size (MB):', size/1e6)
        os.remove('temp.p')
        return size/1e6