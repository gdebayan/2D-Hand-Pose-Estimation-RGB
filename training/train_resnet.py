import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

import sys
sys.path.append("../")

from utils.dataset_resnet import FreiHAND_Resnet
from utils.model import ShallowUNet
from utils.model_resnet import ResnetRegressor
from utils.trainer import Trainer
from utils.prune_utils import PruneUtils

from utils.prep_utils import (
    blur_heatmaps,
    IoULoss,
    COLORMAP,
    N_KEYPOINTS,
    N_IMG_CHANNELS,
    get_norm_params,
    show_data,
)

N_KEYPOINTS = 21
N_IMG_CHANNELS = 3
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 224
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]
MODEL_NEURONS = 16


config = {
    "data_dir": "../data/",
    "epochs": 1000,
    "batch_size": 48,
    "batches_per_epoch": 50,
    "batches_per_epoch_val": 20,
    "learning_rate": 0.1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


train_dataset = FreiHAND_Resnet(config=config, set_type="train")
train_dataloader = DataLoader(
    train_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)

val_dataset = FreiHAND_Resnet(config=config, set_type="val")
val_dataloader = DataLoader(
    val_dataset, config["batch_size"], shuffle=True, drop_last=True, num_workers=2
)


model = ResnetRegressor(N_IMG_CHANNELS, N_KEYPOINTS)
model = model.to(config["device"])

print(model)

# criterion = IoULoss()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
)

ckpt_save_path = f"../checkpoints_resnet"
model_type='resnet'
trainer = Trainer(model, criterion, optimizer, config, ckpt_save_path, scheduler, model_type)
model = trainer.train(train_dataloader, val_dataloader)