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
from utils.prune_utils import QuantizationUtils


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

ckpt = torch.load('../checkpoints/epoch_193', map_location=torch.device(config["device"]))
model.load_state_dict(
    ckpt['model_state_dict'])

model.eval()
print(model)
print("Model loaded")

model_size_pre_quantize = QuantizationUtils.get_size_of_model(model, "pre_quantized")
print("model_size_pre_quantize", model_size_pre_quantize)

# model_quantized_dynamic = QuantizationUtils.dynamic_quantization(model)

# model_size_quantize_dynamic = QuantizationUtils.get_size_of_model(model_quantized_dynamic, "dynamic_quantized")
# print("model_size_quantize_dynamic", model_size_quantize_dynamic)

model_quantized_static = QuantizationUtils.static_quantize_model(model, test_dataloader)

model_size_quantize_static = QuantizationUtils.get_size_of_model(model_quantized_static, "static_quantized")
print("model_size_quantize_static", model_size_quantize_static)

error, exec_time_avg = Evaluator.inference_fwd_baseline(model_quantized_static, test_dataloader)

print("error", error)
print("exec_time_avg", exec_time_avg)

