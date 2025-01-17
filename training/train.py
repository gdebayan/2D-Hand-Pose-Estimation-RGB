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


config = {
    "data_dir": "../data/",
    "epochs": 1000,
    "batch_size": 48,
    "batches_per_epoch": 50,
    "batches_per_epoch_val": 20,
    "learning_rate": 0.1,
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


model = ShallowUNet(N_IMG_CHANNELS, N_KEYPOINTS)
model = model.to(config["device"])

criterion = IoULoss()
optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
)


prune_cls = PruneUtils()
# trainer = Trainer(model, criterion, optimizer, config, scheduler)
# model = trainer.train(train_dataloader, val_dataloader, '/home/ubuntu/project/2D-Hand-Pose-Estimation-RGB/checkpoints/epoch_1') 


#         prune.l1_unstructured(module, name='weight', amount=0.2)
#         prune.remove(module, 'weight')

# model_sparse = prune_cls.apply_sparsity_layer_wise(model=model, sparsity_level=0.2, prune_type='l1', permanent_prune_remove=True)

model_sparse = prune_cls.apply_sparsity_global(model=model, 
                                               sparsity_level=0.2, 
                                               prune_type='l1_unstructred', 
                                               permanent_prune_remove=True)

print('--named buffers --', dict(model.named_buffers()).keys())
print("\n\n\n")
print('--named_parameters --', dict(model.named_parameters()).keys())

print("weig",model_sparse.conv_down1.double_conv[1].weight)