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
        for name, module in model.named_modules():
            for module_type in module_types:
                if isinstance(module, module_type):
                    self.apply_sparsity_module(module=module, prune_type=prune_type, sparsity_level=sparsity_level)
                    if permanent_prune_remove:
                        prune.remove(module, 'weight')
        return model

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
                                  pruning_method=prune.L1Unstructured,
                                  amount=sparsity_level)

        if permanent_prune_remove:
              for p in parameters_to_prune:
                # p takes the form (module, 'weight')
                prune.remove(*p)

        return model

