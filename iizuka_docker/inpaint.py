"""
THIS SCRIPT IS THE PYTORCH IMPLEMENTATION OF THE PAPER:
"Globally and Locally Consistent Image Completion"
WITTEN BY https://github.com/akmtn/pytorch-siggraph2017-inpainting
THAT WE HAVE MODIFIED FOR OUR UNIVERSITY PROJECT
"""


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


# ====================================
# DEALING WITH THE INPUT IMAGE AND MASK
"""
Input image must be a 3-channel RGB image. OpenCV handles JPG, PNG, BMP, etc.
It can accept arbitrary spatial sizes (square, rectangular...), 
with any dimension, even if the model was trained on 256 × 256 patches
"""


# loading input image
input_img = cv2.imread(opt.input)                      # load input image in format: bgr (not rgb!), (H, W, C)
I = torch.from_numpy(cvimg2tensor(input_img)).float()  # from image to tensor

# dealing with the mask
if opt.mask != 'none':  # if the user has provided a mask, we load it
    input_mask = cv2.imread(opt.mask)
    M = torch.from_numpy( 
                cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY) / 255).float() # here we convert the mask to grayscale and normalize it to [0, 1]
    # binarization of the mask:
    M[M <= 0.2] = 0.0
    M[M > 0.2] = 1.0

    # ===================================
    M = 1.0 - M # !!!! added only cause the created masks have 0s in the hole region and 1s in the untouched region, but the model expects the opposite
    # ===================================

    M = M.view(1, M.size(0), M.size(1)) # reshaping the mask to have the same dimensions as the input mask 
    assert I.size(1) == M.size(1) and I.size(2) == M.size(2) # sanity check

else: # here we generate a random mask if the user has not provided one
    # generate random holes
    M = torch.FloatTensor(1, I.size(1), I.size(2)).fill_(0) # Initialize empty mask
    nHoles = np.random.randint(1, 4)   # Between 1 and 3 rectangles
    print(nHoles)
    print('w: ', I.size(2))
    print('h: ', I.size(1))
    for _ in range(nHoles):
        # random hole sizes:
        mask_w = np.random.randint(32, 128)
        mask_h = np.random.randint(32, 128)
        assert I.size(1) > mask_h or I.size(2) > mask_w
        # Random position:
        px = np.random.randint(0, I.size(2)-mask_w)
        py = np.random.randint(0, I.size(1)-mask_h)
        # Fill the hole with 1s:
        M[:, py:py+mask_h, px:px+mask_w] = 1
        # now M shape: (1, H, W)

# channel-wise normalization:
for i in range(3):
    I[i, :, :] = I[i, :, :] - datamean[i]

# make mask_3ch (so from 1-channel to 3-channel mask)
M_3ch = torch.cat((M, M, M), 0)
# Apply mask to image (holes are black, i.e. pixel value = 0)
Im = I * (M_3ch*(-1)+1)

# set up input for the model: concatenate image and mask, and add a batch dimension
input = torch.cat((Im, M), 0)
input = input.view(1, input.size(0), input.size(1), input.size(2)).float()
# so now the input shape is (1, 4, H, W)

# GPU handling
if opt.gpu:
    print('using GPU...')
    model.cuda()
    input = input.cuda()


# ====================================
# MODEL EVALUATION
res = model.forward(input)[0].cpu() # forward pass through the model, and get the output (the inpainted image)

# make out (denormalization)
for i in range(3):
    I[i, :, :] = I[i, :, :] + datamean[i] # now I = original image (restored)
# here we combine the inpainted image with the original image, using the mask to determine which pixels to take from which image
out = res.float()*M_3ch.float() + I.float()*(M_3ch*(-1)+1).float()

# post-processing
if opt.postproc:
    print('post-postprocessing...')
    target = input_img    # background
    source = tensor2cvimg(out.numpy())    # foreground
    mask = input_mask
    out = blend(target, source, mask, offset=(0, 0))

    out = torch.from_numpy(cvimg2tensor(out))


# save images
print('save images...')
vutils.save_image(out, 'out.png', normalize=True)
# vutils.save_image(Im, 'masked_input.png', normalize=True)
# vutils.save_image(M_3ch, 'mask.png', normalize=True)
# vutils.save_image(res, 'res.png', normalize=True)
print('Done')
