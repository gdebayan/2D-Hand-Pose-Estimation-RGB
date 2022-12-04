import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append("../")

from utils.prep_utils import (
    COLORMAP,
    heatmaps_to_coordinates,
    N_KEYPOINTS,
    RAW_IMG_SIZE,
    MODEL_IMG_SIZE,
    show_batch_predictions,
    DATASET_MEANS,
    DATASET_STDS,
)
from utils.model import ShallowUNet
from utils.dataset import FreiHAND
from utils.evaluator import Evaluator
from utils.prune_utils import PruneUtils, SparsityCalculator


config = {
    "data_dir": "../data/",
    "model_path": "weights/model_final",
    "test_batch_size": 1,
    "device": "cpu"
}

test_dataset = FreiHAND(config=config, set_type="test")
test_dataloader = DataLoader(
    test_dataset,
    config["test_batch_size"],
    shuffle=True,
    drop_last=False,
    num_workers=2,
)

# def sparsify_model()


model = ShallowUNet(3, 21)

# MODEL_PATH='../checkpoints/epoch_193'
# ckpt = torch.load(MODEL_PATH, map_location=torch.device(config["device"]))
# model.load_state_dict(
#     ckpt['model_state_dict'])


MODEL_PATH='../checkpoints/global_l1_no_rewind/prune_iter_13'
model = torch.load(MODEL_PATH)

model.eval()
# print(model)
print("Model loaded")

prune_utils = PruneUtils()
model_sparse_layers = prune_utils._get_module_layers_for_global_pruning(model)
prune_utils._apply_prune_remove(parameters_to_prune=model_sparse_layers)

# for k, v in model.named_parameters():
#     print("k", k, 'v', v.to_sparse())

sd = model.state_dict()

for param_tensor in sd:
    sd[param_tensor] = model.state_dict()[param_tensor].to_sparse()
    # print(param_tensor, "\t", model.state_dict()[param_tensor])

# model.load_state_dict(sd)


model_sparse_pruned  = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

# print(model_sparse_pruned.state_dict())

error, exec_time_avg = Evaluator.inference_fwd_baseline(model_sparse_pruned, test_dataloader)

# print("error", error)
# print("exec_time_avg", exec_time_avg)

