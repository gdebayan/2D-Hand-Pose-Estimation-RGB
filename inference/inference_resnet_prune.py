import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append("../")

from utils.model_resnet import ResnetRegressor
from utils.dataset_resnet import FreiHAND_Resnet
from utils.evaluator import Evaluator
from utils.prune_utils import PruneUtils, SparsityCalculator



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


# model = ResnetRegressor(3, 21)
# ckpt = torch.load('../checkpoints_resnet/epoch_125', map_location=torch.device(config["device"]))
# model.load_state_dict(
#     ckpt['model_state_dict'])

MODEL_PATH='../checkpoints_resnet/global_l1_no_rewind/prune_iter_5'
model = torch.load(MODEL_PATH)
model = model.to("cpu")

model.eval()
print(model)
print("Model loaded")

prune_utils = PruneUtils()
model_sparse_layers = prune_utils._get_module_layers_for_global_pruning(model)
prune_utils._apply_prune_remove(parameters_to_prune=model_sparse_layers)

# sd = model.state_dict()

# for param_tensor in sd:
#     sd[param_tensor] = model.state_dict()[param_tensor].to_sparse()
# # model.load_state_dict(sd)


model_sparse_pruned  = torch.quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )

error, exec_time_avg = Evaluator.inference_fwd_resnet(model_sparse_pruned, test_dataloader)

print("error", error)
print("exec_time_avg", exec_time_avg)

