import torch.nn as nn
import torch.nn.utils.prune as prune

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
        elif prune_type == 'l2':
            prune.ln_structured(module, name="weight", amount=sparsity_level, n=2, dim=0)
        else:
            raise Exception(f"Unsupported Pruning Type {prune_type}")

    def apply_sparsity_layer_wise(self, model, 
                                        sparsity_level, 
                                        prune_type,
                                        module_types=[nn.Linear, nn.Conv2d],
                                        permanent_prune_remove=False):
        for name, module in model.named_modules():
            for module_type in module_types:
                if isinstance(module, module_type):
                    self.apply_sparsity_module(module=module, prune_type=prune_type, sparsity_level=sparsity_level)
                    if permanent_prune_remove:
                        prune.remove(module, 'weight')
        return model


