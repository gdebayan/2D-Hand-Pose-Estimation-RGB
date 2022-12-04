import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

from utils.prep_utils import MODEL_NEURONS



class ResnetRegressor(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.model_res = models.resnet50(pretrained=True)
        # self.model_res.fc = nn.Linear(512, out_channel*2)
        self.proj_layer = nn.Linear(1000, out_channel*2)


    def forward(self, x):
        out = self.model_res(x)
        # print('out shape', out.shape)
        out = self.proj_layer(out)
        return out