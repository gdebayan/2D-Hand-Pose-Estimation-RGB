import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

import sys
import os
sys.path.append("../")

from utils.dataset import FreiHAND
from utils.model import ShallowUNet
from utils.trainer import Trainer
from utils.prune_utils import PruneUtils, SparsityCalculator

from utils.prep_utils import (
    blur_heatmaps,
    IoULoss,
    COLORMAP,
    N_KEYPOINTS,
    N_IMG_CHANNELS,
    get_norm_params,
    show_data,
)


config = {
    "data_dir": "../data/",
    "epochs": 1,
    "batch_size": 48,
    "batches_per_epoch": 50,
    "batches_per_epoch_val": 20,
    "learning_rate": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

##################################
# Train/Val Dataset/ Dataloaders #
##################################

train_dataset = FreiHAND(config=config, set_type="train")
train_dataloader = DataLoader(
    train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)

val_dataset = FreiHAND(config=config, set_type="val")
val_dataloader = DataLoader(
    val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)

##################################
# Load Baseline Model Checkpoint #
##################################

BASE_CKPT_PATH = '../checkpoints/epoch_193'

def load_model_ckpt(model, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

model = ShallowUNet(N_IMG_CHANNELS, N_KEYPOINTS)
model = model.to(config["device"])
model = load_model_ckpt(model, BASE_CKPT_PATH)
criterion = IoULoss()
scheduler=None

######################
# Load Pruning Tools #
######################

prune_cls = PruneUtils()

PRUNE_ITERATIONS = 20

PRUNE_TYPE = 'global_l1_with_rewind'
CKPT_SAVE_PATH=f'../checkpoints/{PRUNE_TYPE}'
os.makedirs(CKPT_SAVE_PATH, exist_ok=True)

###################
# Perform Pruning #
###################

for prune_iter in range(0, PRUNE_ITERATIONS):

    prev_iter_ckpt = f'{CKPT_SAVE_PATH}/prune_iter_{prune_iter - 1}'
    if os.path.exists(prev_iter_ckpt):
        model = torch.load(prev_iter_ckpt)

    model, pruned_params_list = prune_cls.apply_sparsity_global(model=model, 
                                                                sparsity_level=0.2, 
                                                                prune_type='l1_unstructred', 
                                                                permanent_prune_remove=False)

    layer_wise_sparsity, tot_sparsity, tot_sparsity_pruned_layers = SparsityCalculator.calculate_sparsity_pruned_model(model)

    # Rewind, load the previous weights
    model = prune_cls._rewind_load_prev_weights(model, BASE_CKPT_PATH)

    # Save model for future iteration
    save_path = f'{CKPT_SAVE_PATH}/prune_iter_{prune_iter}'
    torch.save(model, save_path)

    # Remove weight_orig and weight_mask --> make weight permanent 
    prune_cls._apply_prune_remove(parameters_to_prune=pruned_params_list)

    # Calculate Size after "sparsifying" the weights
    size_mb = prune_cls.calculate_model_size(model)

    print("tot_sparsity", tot_sparsity, "tot_sparsity_pruned_layers", tot_sparsity_pruned_layers, "size_mb", size_mb)

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    trainer = Trainer(model, criterion, optimizer, config)
    model = trainer.train(train_dataloader, val_dataloader) 



"""
(pytorch) ubuntu@ip-172-31-85-126:~/project/2D-Hand-Pose-Estimation-RGB/training$ python3 train_baseline_global_l1_imp_rewind.py 
14.194285 MB
tot_sparsity 0.19934609919259944 tot_sparsity_pruned_layers 0.2 size_mb 14.194285
Epoch: 1/1, Train Loss=0.41406552, Val Loss=0.4658777998
11.374061 MB
tot_sparsity 0.358822978546679 tot_sparsity_pruned_layers 0.36 size_mb 11.374061
Epoch: 1/1, Train Loss=0.4235164792, Val Loss=0.4672904525
9.118125 MB
tot_sparsity 0.4864048891910929 tot_sparsity_pruned_layers 0.48800040849673204 size_mb 9.118125
Epoch: 1/1, Train Loss=0.4432652651, Val Loss=0.4766097267
7.313389 MB
tot_sparsity 0.5884700105454738 tot_sparsity_pruned_layers 0.5904003267973856 size_mb 7.313389
Epoch: 1/1, Train Loss=0.4714849299, Val Loss=0.4810423454
5.869421 MB
tot_sparsity 0.6701221076289785 tot_sparsity_pruned_layers 0.6723202614379085 size_mb 5.869421
Epoch: 1/1, Train Loss=0.5018632576, Val Loss=0.4982609493
4.714285 MB
tot_sparsity 0.7354429709734815 tot_sparsity_pruned_layers 0.7378553921568628 size_mb 4.714285
Epoch: 1/1, Train Loss=0.5318483453, Val Loss=0.5287743409
3.790125 MB
tot_sparsity 0.7877000688102344 tot_sparsity_pruned_layers 0.7902839052287581 size_mb 3.790125
Epoch: 1/1, Train Loss=0.5754411746, Val Loss=0.5436701491
3.050861 MB
tot_sparsity 0.8295053399184863 tot_sparsity_pruned_layers 0.8322263071895425 size_mb 3.050861
Epoch: 1/1, Train Loss=0.6000723745, Val Loss=0.5670539084
2.459437 MB
tot_sparsity 0.8629495568050879 tot_sparsity_pruned_layers 0.86578022875817 size_mb 2.459437
Epoch: 1/1, Train Loss=0.6408820386, Val Loss=0.5737730548
1.986349 MB
tot_sparsity 0.8897061517978201 tot_sparsity_pruned_layers 0.892624591503268 size_mb 1.986349
Epoch: 1/1, Train Loss=0.6744297918, Val Loss=0.6042165699
1.607917 MB
tot_sparsity 0.9111106134697051 tot_sparsity_pruned_layers 0.9140992647058823 size_mb 1.607917
Epoch: 1/1, Train Loss=0.7151709409, Val Loss=0.6648958411
1.304941 MB
tot_sparsity 0.9282337756460629 tot_sparsity_pruned_layers 0.9312785947712419 size_mb 1.304941
^Z
[4]+  Stopped                 python3 train_baseline_global_l1_imp_rewind.py
(pytorch) ubuntu@ip-172-31-85-126:~/project/2D-Hand-Pose-Estimation-RGB/training$ 
"""
