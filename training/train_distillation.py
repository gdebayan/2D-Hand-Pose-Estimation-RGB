import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as prune

import sys
sys.path.append("../")

from utils.dataset import FreiHAND
from utils.model import ShallowUNet, StudentShallowNet
from utils.trainer_model_distillation import TrainerDistillation
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
    "epochs": 200,
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

test_dataset = FreiHAND(config=config, set_type="test")
test_dataloader = DataLoader(
    test_dataset,
    config["batch_size"],
    shuffle=True,
    drop_last=False,
    num_workers=2,
)


model = StudentShallowNet(N_IMG_CHANNELS, N_KEYPOINTS)
model = model.to(config["device"])


TEACHER_MODEL_PATH = '../checkpoints/epoch_193'
teacher_model = ShallowUNet(N_IMG_CHANNELS, N_KEYPOINTS)
teacher_ckpt = torch.load(TEACHER_MODEL_PATH)
teacher_model.load_state_dict(teacher_ckpt["model_state_dict"])


distill_criterion = IoULoss() #nn.MSELoss()
student_criterion = IoULoss()
alpha_loss = 0.2

optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer, factor=0.5, patience=20, verbose=True, threshold=0.00001
)

ckpt_save_path = f'../checkpoints_complex_model_distilation_iou_criterion_{alpha_loss}_lr{config["learning_rate"]}'

trainer = TrainerDistillation(model, 
                              teacher_model,
                              distill_criterion, 
                              student_criterion,
                              alpha_loss,
                              optimizer, 
                              config, 
                              ckpt_save_path, 
                              scheduler)

# trainer = Trainer(model, criterion, optimizer, config, scheduler)
model = trainer.train(train_dataloader, val_dataloader, test_dataloader) 

