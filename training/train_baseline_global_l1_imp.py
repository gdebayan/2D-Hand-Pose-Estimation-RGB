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
    "learning_rate": 0.01,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


train_dataset = FreiHAND(config=config, set_type="train")
train_dataloader = DataLoader(
    train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)

val_dataset = FreiHAND(config=config, set_type="val")
val_dataloader = DataLoader(
    val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)

def load_model_ckpt(model):
    BASE_CKPT_PATH = '../checkpoints/epoch_193'
    checkpoint = torch.load(BASE_CKPT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


model = ShallowUNet(N_IMG_CHANNELS, N_KEYPOINTS)
model = model.to(config["device"])
model = load_model_ckpt(model)
criterion = IoULoss()
scheduler=None

prune_cls = PruneUtils()

PRUNE_ITERATIONS = 20

PRUNE_TYPE = 'global_l1'
CKPT_SAVE_PATH=f'../checkpoints/{PRUNE_TYPE}'
os.makedirs(CKPT_SAVE_PATH, exist_ok=True)

for prune_iter in range(0, PRUNE_ITERATIONS):

    prev_iter_ckpt = f'{CKPT_SAVE_PATH}/prune_iter_{prune_iter - 1}'
    if os.path.exists(prev_iter_ckpt):
        model = torch.load(prev_iter_ckpt)

    model, pruned_params_list = prune_cls.apply_sparsity_global(model=model, 
                                                                sparsity_level=0.2, 
                                                                prune_type='l1_unstructred', 
                                                                permanent_prune_remove=False)

    layer_wise_sparsity, tot_sparsity, tot_sparsity_pruned_layers = SparsityCalculator.calculate_sparsity_pruned_model(model)


    save_path = f'{CKPT_SAVE_PATH}/prune_iter_{prune_iter}'
    torch.save(model, save_path)

    prune_cls._apply_prune_remove(parameters_to_prune=pruned_params_list)
    size_mb = prune_cls.calculate_model_size(model)

    print("tot_sparsity", tot_sparsity, "tot_sparsity_pruned_layers", tot_sparsity_pruned_layers, "size_mb", size_mb)

    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    trainer = Trainer(model, criterion, optimizer, config)
    model = trainer.train(train_dataloader, val_dataloader) 











