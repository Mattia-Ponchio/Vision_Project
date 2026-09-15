import argparse
import os
import torch
# from torch.legacy import nn
# from torch.legacy.nn.Sequential import Sequential
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import torchvision.utils as vutils
from utils import *
from poissonblending import prepare_mask, blend


# ====================================
# INITIAL OPTIONS FOR RUNNING THE MODEL
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='none', help='Input image')
parser.add_argument('--mask', default='none', help='Mask image')
parser.add_argument('--model_path', 
                    default='completionnet_places2.t7',     # you can change here which weights to use
                    help='Trained model')
parser.add_argument('--gpu', default=False, action='store_true',
                    help='use GPU')
parser.add_argument('--postproc', default=False, action='store_true',
                    help='Disable post-processing')
opt = parser.parse_args()
print(opt)


# ====================================
# NETWORK LOADING
data = load_lua(opt.model_path)   # here we load the weights (and the architecture)
model = data.model                
model.evaluate()                  # switches the model to inference/evaluation mode, so disables dropout, fixes batchnorm behavior etc
datamean = data.mean              # the model expects normalized input



print(len(model.modules))

for i, m in enumerate(model.modules[:10]):
    print(i, type(m))


def count_parameters_legacy(model):
    total = 0

    for module in model.modules:   # ← NO parentheses
        
        if hasattr(module, 'weight') and module.weight is not None:
            total += module.weight.numel()
        
        if hasattr(module, 'bias') and module.bias is not None:
            total += module.bias.numel()

    return total


total_params = count_parameters_legacy(model)
print("Total parameters:", total_params)


def count_all_params(model):
    total = 0

    for module in model.modules:
        for attr in ['weight', 'bias', 'running_mean', 'running_var']:
            if hasattr(module, attr):
                val = getattr(module, attr)
                if val is not None:
                    total += val.numel()

    return total


print("All params (incl buffers):", count_all_params(model))



for i, m in enumerate(model.modules):
    if hasattr(m, 'weight') and m.weight is not None:
        print(i, m.__class__.__name__, m.weight.size())