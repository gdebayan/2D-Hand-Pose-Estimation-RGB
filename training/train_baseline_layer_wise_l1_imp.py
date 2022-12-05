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
from utils.evaluator import Evaluator


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
    'test_batch_size': 1,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

test_dataset = FreiHAND(config=config, set_type="test")
test_dataloader = DataLoader(
    test_dataset,
    config["test_batch_size"],
    shuffle=True,
    drop_last=False,
    num_workers=2,
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

PRUNE_ITERATIONS = 10

PRUNE_TYPE = 'baseline_layer_wise_l1_no_rewind'
CKPT_SAVE_PATH=f'../checkpoints/{PRUNE_TYPE}'
os.makedirs(CKPT_SAVE_PATH, exist_ok=True)

###################
# Perform Pruning #
###################

info_list = []

for prune_iter in range(0, PRUNE_ITERATIONS):

    prev_iter_ckpt = f'{CKPT_SAVE_PATH}/prune_iter_{prune_iter - 1}'
    if os.path.exists(prev_iter_ckpt):
        model = torch.load(prev_iter_ckpt)

    model, pruned_params_list = prune_cls.apply_sparsity_layer_wise(model=model, 
                                                                sparsity_level=0.2, 
                                                                prune_type='l1', 
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

    rmse, exec_time_avg = Evaluator.inference_fwd_baseline(model, test_dataloader)

    info_list.append({'prune_iter': prune_iter,
                      'tot_sparsity': tot_sparsity, 
                      'tot_sparsity_pruned_layers': tot_sparsity_pruned_layers,
                      'size_mb': size_mb,
                      'iou_train_loss': trainer.loss['train'][-1],
                      'iou_val_loss': trainer.loss['val'][-1],
                      'test_rmse': rmse,
                      'exec_time_avg': exec_time_avg})
    print(prune_iter, "rmse", rmse, "tot_sparsity", tot_sparsity, "tot_sparsity_pruned_layers", tot_sparsity_pruned_layers, "size_mb", size_mb)

import pandas as pd
df = pd.DataFrame(info_list)
# df.head()


df.to_csv(f'{CKPT_SAVE_PATH}/{PRUNE_TYPE}.csv') 


"""
14.194285 MB
tot_sparsity 0.19934609919259944 tot_sparsity_pruned_layers 0.2 size_mb 14.194285
Epoch: 1/1, Train Loss=0.4025208085, Val Loss=0.433343683
11.374061 MB
tot_sparsity 0.358822978546679 tot_sparsity_pruned_layers 0.36 size_mb 11.374061
Epoch: 1/1, Train Loss=0.4093093311, Val Loss=0.4414410591
9.118125 MB
tot_sparsity 0.4864048891910929 tot_sparsity_pruned_layers 0.48800040849673204 size_mb 9.118125
Epoch: 1/1, Train Loss=0.4347823671, Val Loss=0.4576438013
7.313389 MB
tot_sparsity 0.5884700105454738 tot_sparsity_pruned_layers 0.5904003267973856 size_mb 7.313389
Epoch: 1/1, Train Loss=0.4650907785, Val Loss=0.4727333444
5.869421 MB
tot_sparsity 0.6701221076289785 tot_sparsity_pruned_layers 0.6723202614379085 size_mb 5.869421
Epoch: 1/1, Train Loss=0.5100779533, Val Loss=0.5023046505
4.714285 MB
tot_sparsity 0.7354429709734815 tot_sparsity_pruned_layers 0.7378553921568628 size_mb 4.714285
Epoch: 1/1, Train Loss=0.5546856745, Val Loss=0.5295991784
3.790125 MB
tot_sparsity 0.7877000688102344 tot_sparsity_pruned_layers 0.7902839052287581 size_mb 3.790125
Epoch: 1/1, Train Loss=0.607531641, Val Loss=0.5591910765
3.050861 MB
tot_sparsity 0.8295053399184863 tot_sparsity_pruned_layers 0.8322263071895425 size_mb 3.050861
Epoch: 1/1, Train Loss=0.6399999541, Val Loss=0.5779062793
2.459437 MB
tot_sparsity 0.8629495568050879 tot_sparsity_pruned_layers 0.86578022875817 size_mb 2.459437
Epoch: 1/1, Train Loss=0.6804971298, Val Loss=0.6193254278
1.986349 MB
tot_sparsity 0.8897061517978201 tot_sparsity_pruned_layers 0.892624591503268 size_mb 1.986349
Epoch: 1/1, Train Loss=0.7153971499, Val Loss=0.6418271945
1.607917 MB
tot_sparsity 0.9111106134697051 tot_sparsity_pruned_layers 0.9140992647058823 size_mb 1.607917
Epoch: 1/1, Train Loss=0.7418540634, Val Loss=0.6700074786
1.304941 MB
tot_sparsity 0.9282337756460629 tot_sparsity_pruned_layers 0.9312785947712419 size_mb 1.304941
Epoch: 1/1, Train Loss=0.755508805, Val Loss=0.7213247731
1.062701 MB
tot_sparsity 0.9419327125482995 tot_sparsity_pruned_layers 0.9450224673202614 size_mb 1.062701
Epoch: 1/1, Train Loss=0.7759486133, Val Loss=0.7123488841
0.869037 MB
tot_sparsity 0.9528914549089385 tot_sparsity_pruned_layers 0.9560171568627451 size_mb 0.869037
"""