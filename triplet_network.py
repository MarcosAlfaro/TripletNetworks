"""
THIS PROGRAM CONTAINS A CLASS THAT DEFINES THE NETWORK ARCHITECTURE
Only "train_coarse_loc.py" and "train_global_loc.py" call this program
"""


import torch.nn as nn
import torch
from torchvision.models import VGG16_Weights

vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.DEFAULT)


class TripletNetwork(nn.Module):

    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn1 = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        self.fc1 = nn.Sequential(
            nn.Linear(4 * 16 * 512, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5))

    def forward_once(self, x):
        verbose = False

        if verbose:
            print("Input: ", x.size())

        out = self.cnn1(x)

        if verbose:
            print("Output matricial: ", out.size())

        out = self.avgpool(out)
        if verbose:
            print("Output avgpool: ", out.size())
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        norm = True
        if norm:
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out

    def forward(self, input1):
        out1 = self.forward_once(input1)
        return out1
