import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("../")

from utils.model_resnet import ResnetRegressor
from utils.dataset_resnet import FreiHAND_Resnet
from utils.evaluator import Evaluator


config = {
    "data_dir": "../data/",
    "model_path": "weights/model_final",
    "test_batch_size": 1,
    "device": "cpu"
}

test_dataset = FreiHAND_Resnet(config=config, set_type="test")
test_dataloader = DataLoader(
    test_dataset,
    config["test_batch_size"],
    shuffle=True,
    drop_last=False,
    num_workers=2,
)


model = ResnetRegressor(3, 21)
ckpt = torch.load('../checkpoints_resnet/epoch_125', map_location=torch.device(config["device"]))
model.load_state_dict(
    ckpt['model_state_dict'])

model.eval()
print(model)
print("Model loaded")

error, exec_time_avg = Evaluator.inference_fwd_resnet(model, test_dataloader)

print("error", error)
print("exec_time_avg", exec_time_avg)

