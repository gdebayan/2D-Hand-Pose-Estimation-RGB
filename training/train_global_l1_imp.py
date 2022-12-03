import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

import sys
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

PRUNE_ITERATIONS = 1
CKPT_SAVE_PATH='ckpt_save_path'

PRUNE_TYPE = 'global_l1'
for prune_iter in range(0, PRUNE_ITERATIONS):

    model = prune_cls.apply_sparsity_global(model=model, 
                                            sparsity_level=0.2, 
                                            prune_type='l1_unstructred', 
                                            permanent_prune_remove=False)

    layer_wise_sparsity, tot_sparsity, tot_sparsity_pruned_layers = SparsityCalculator.calculate_sparsity_pruned_model(model)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    # print("layer_wise_sparsity", layer_wise_sparsity, "tot_sparsity", tot_sparsity, "tot_sparsity_pruned_layers", tot_sparsity_pruned_layers)
    print("tot_sparsity", tot_sparsity, "tot_sparsity_pruned_layers", tot_sparsity_pruned_layers)

    ckpt_save_path=f'../checkpoints/{PRUNE_TYPE}/iter_{prune_iter}'
    trainer = Trainer(model, criterion, optimizer, config, ckpt_save_path)
    model = trainer.train(train_dataloader, val_dataloader) 











