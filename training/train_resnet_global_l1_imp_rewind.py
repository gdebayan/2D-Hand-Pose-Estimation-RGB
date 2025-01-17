import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

import sys
import os
sys.path.append("../")

from utils.dataset_resnet import FreiHAND_Resnet
from utils.model_resnet import ResnetRegressor
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
    "learning_rate": 0.01,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

##################################
# Train/Val Dataset/ Dataloaders #
##################################

train_dataset = FreiHAND_Resnet(config=config, set_type="train")
train_dataloader = DataLoader(
    train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)

val_dataset = FreiHAND_Resnet(config=config, set_type="val")
val_dataloader = DataLoader(
    val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)

##################################
# Load Baseline Model Checkpoint #
##################################
BASE_CKPT_PATH = '../checkpoints_resnet/epoch_126'

def load_model_ckpt(model, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

model = ResnetRegressor(N_IMG_CHANNELS, N_KEYPOINTS)
model = model.to(config["device"])
model = load_model_ckpt(model, BASE_CKPT_PATH)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
)

scheduler=None

######################
# Load Pruning Tools #
######################

prune_cls = PruneUtils()

PRUNE_ITERATIONS = 20

PRUNE_TYPE = 'global_l1_with_rewind'
CKPT_SAVE_PATH=f'../checkpoints_resnet/{PRUNE_TYPE}'
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

    save_path = f'{CKPT_SAVE_PATH}/prune_iter_{prune_iter}'
    torch.save(model, save_path)

    prune_cls._apply_prune_remove(parameters_to_prune=pruned_params_list)
    size_mb = prune_cls.calculate_model_size(model)

    print("tot_sparsity", tot_sparsity, "tot_sparsity_pruned_layers", tot_sparsity_pruned_layers, "size_mb", size_mb)

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    model_type='resnet'
    ckpt_save_path=None

    trainer = Trainer(model, criterion, optimizer, config, ckpt_save_path, scheduler, model_type)
    model = trainer.train(train_dataloader, val_dataloader) 

