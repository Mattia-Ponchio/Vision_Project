import argparse
import os
import time
import torch
import cv2
import numpy as np
from torch.utils.serialization import load_lua
import torchvision.utils as vutils

from utils import *
from poissonblending import blend


# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, default='test_images', help='Folder containing input images')
parser.add_argument('--mask_dir', type=str, default='masks', help='Folder containing masks')
parser.add_argument('--output_dir', type=str, default='results', help='Where to save results')
parser.add_argument('--model_path', type=str, default='completionnet_places2.t7', help='Trained model path')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
parser.add_argument('--postproc', action='store_true', help='Apply poisson blending postprocessing')
opt = parser.parse_args()
print(opt)


# ============================================================
# LOAD MODEL ONCE (IMPORTANT)
# ============================================================
data = load_lua(opt.model_path)     # here we load the weights (and the architecture)
model = data.model
model.evaluate()                    # switches the model to inference/evaluation mode, so disables dropout, fixes batchnorm behavior etc
datamean = data.mean                # the model expects normalized input

if opt.gpu:
    print("Using GPU...")
    model.cuda()

# create output folder
if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir)


# ============================================================
# LOAD FILE LISTS
# ============================================================
image_list = sorted([f for f in os.listdir(opt.input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

mask_list = sorted([f for f in os.listdir(opt.mask_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

tot_iterations = len(image_list) * len(mask_list)  # just for debug


print("Found {} images and {} masks".format(len(image_list), len(mask_list)))


# ============================================================
# MAIN LOOP: IMAGE × MASK
# ============================================================
it = 0 # debug
start_time = time.time()  # debug

for img_name in image_list:
    img_path = os.path.join(opt.input_dir, img_name)

    # load image
    input_img = cv2.imread(img_path)
    if input_img is None:
        print("Skipping corrupted image:", img_name)
        continue

    I0 = torch.from_numpy(cvimg2tensor(input_img)).float()       # from image to tensor

    for mask_name in mask_list:
        mask_path = os.path.join(opt.mask_dir, mask_name)

        # ====================================================
        # LOAD MASK
        # ====================================================
        input_mask = cv2.imread(mask_path, 0)  # grayscale

        if input_mask is None:
            print("Skipping corrupted mask:", mask_name)
            continue

        M = torch.from_numpy(input_mask / 255.0).float()

        # binarize
        M[M <= 0.2] = 0
        M[M > 0.2] = 1

        # IMPORTANT: invert mask (model convention)
        M = 1.0 - M

        # reshape
        M = M.view(1, M.size(0), M.size(1))

        # check size match
        if I0.size(1) != M.size(1) or I0.size(2) != M.size(2):
            print("Skipping due to size mismatch:", img_name, mask_name)
            continue

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================
        I = I0.clone()

        for i in range(3):
            I[i] = I[i] - datamean[i]

        # mask to 3 channels
        M3 = torch.cat((M, M, M), 0)

        # apply mask
        Im = I * (1 - M3)

        # model input
        inp = torch.cat((Im, M), 0)
        inp = inp.view(1, inp.size(0), inp.size(1), inp.size(2)).float()

        if opt.gpu:
            inp = inp.cuda()

        # ====================================================
        # FORWARD PASS
        # ====================================================
        with torch.no_grad():
            res = model.forward(inp)[0].cpu()

        # restore image
        I_restored = I.clone()
        for i in range(3):
            I_restored[i] = I_restored[i] + datamean[i]

        out = res * M3 + I_restored * (1 - M3)

        # ====================================================
        # POSTPROCESS (OPTIONAL)
        # ====================================================
        if opt.postproc:
            target = input_img.copy()
            source = tensor2cvimg(out.numpy())
            out_np = blend(target, source, input_mask, offset=(0, 0))
            out = torch.from_numpy(cvimg2tensor(out_np))

        # ====================================================
        # SAVE OUTPUT
        # ====================================================
        img_base = os.path.splitext(img_name)[0]
        mask_base = os.path.splitext(mask_name)[0]

        out_name = "{}__{}.png".format(img_base, mask_base)
        out_path = os.path.join(opt.output_dir, out_name)

        vutils.save_image(out, out_path, normalize=True)

        it += 1
        print("Processed {}/{}: {} ".format(it, tot_iterations, out_path))

final_time = time.time() - start_time
print("Total processing time: {:.0f}m{:.0f}sec".format(final_time // 60, final_time % 60))
print("Done.")