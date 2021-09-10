import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
from tracking_backbone.alexnet import AlexNet
#import pytorch_colors as colors
import numpy as np

from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        self.features = AlexNet().cuda()
        self.features.load_state_dict(torch.load('tracking_backbone/alexnet-bn.pth'))
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        f1, f2, f3, f4, f5 = self.features(x)
        m1, m2, m3, m4, m5 = self.features(y)
        # loss1 = torch.mean(torch.pow(f1-m1, 2))
        # loss2 = torch.mean(torch.pow(f2-m2, 2))
        loss3 = torch.mean(torch.pow(f3-m3, 2))
        loss4 = torch.mean(torch.pow(f4-m4, 2))
        loss5 = torch.mean(torch.pow(f5-m5, 2))
        loss = loss3 + loss4 +loss5
        return loss