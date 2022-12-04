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

MODEL_PATH='../checkpoints/epoch_193'
MODEL_PATH='../checkpoints/global_l1_no_rewind/prune_iter_13'

ckpt = torch.load('../checkpoints/epoch_193', map_location=torch.device(config["device"]))
model.load_state_dict(
    ckpt['model_state_dict'])

model.eval()
print(model)
print("Model loaded")

model_pruned  = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

error, exec_time_avg = Evaluator.inference_fwd_baseline(model_pruned, test_dataloader)

print("error", error)
print("exec_time_avg", exec_time_avg)

