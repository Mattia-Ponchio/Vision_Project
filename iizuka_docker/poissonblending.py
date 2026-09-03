#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module implements Poisson blending, a technique for blending an image region into another image. 
The main function `blend` takes a target image (the original image), a source image (the generated image), a mask defining the region to blend, 
and an optional offset for positioning the source image within the target image. 
The blending is performed using Poisson equations to ensure smooth transitions between the images.
It basically tries to minimize the difference between the gradients of the source and target images in the blended region, 
resulting in a seamless blend.
"""

import numpy as np
import scipy.sparse
import cv2
import pyamg

# pre-process the mask array so that uint64 types from opencv.imread can be adapted

# original version for masks with 1s in the hole region and 0s in the untouched region
"""
def prepare_mask(mask):
    # this function is used to convert a mask image to a binary mask (0 or 1) for Poisson blending

    if type(mask[0][0]) is np.ndarray:  # check if the mask is a 3-channel image (e.g., RGB)
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for i in range(mask.shape[0]):       # cycle through the height of the mask
            for j in range(mask.shape[1]):   # cycle through the width of the mask
                if sum(mask[i][j]) > 0:      # check if the pixel is not black, i.e. different from (0, 0, 0), then make it white (1)
                    result[i][j] = 1
                else:                        # else make it black (0)
                    result[i][j] = 0
        mask = result
    # final result: a 2D array of shape (H, W) with values 0 or 1, where 1 indicates the region to blend and 0 indicates the background
    return mask  
"""

# version for masks with 0s in the hole region and 1s in the untouched region
def prepare_mask(mask):
    """
    Convert mask to binary (0/1) where:   1 = region to blend (hole),   0 = background
    """
    # if mask is 3-channel → convert to grayscale
    if len(mask.shape) == 3:
        mask = mask.sum(axis=2)

    # IMPORTANT: invert logic
    # black (0) → hole → 1
    # white (>0) → background → 0
    mask = (mask == 0).astype(np.uint8)

    return mask



def blend(img_target, img_source, img_mask, offset=(0, 0)):
    """
    This function blends a source image (generated one) into a target image (original one) using Poisson blending.
    """

    # compute regions to be blended based on the offset and the sizes of the images
    # if offset in not (0,0), then we need to crop the source and target images accordingly, so that they have the same size
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[0]-offset[0], img_source.shape[0]),
            min(img_target.shape[1]-offset[1], img_source.shape[1]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[0], img_source.shape[0]+offset[0]),
            min(img_target.shape[1], img_source.shape[1]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask = prepare_mask(img_mask)
    # now we convert the mask to a boolean array, where False indicates the background and True indicates the region to blend
    img_mask[img_mask==0] = False   
    img_mask[img_mask!=False] = True 

    # ===== create coefficient matrix A for the Poisson equation =========
    # create a sparse identity matrix of size (H*W, H*W) where H and W are the height and width of the region to blend
    A = scipy.sparse.identity(np.prod(region_size), format='lil') 
    # cycles over the pixels in the region to blend, and for each pixel that is in the mask (i.e. to be blended), 
    # it uses laplacian operator to set the coefficients in the matrix A
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr() # we obtain a sparse matrix used to solve the Poisson equation Ax = b, 
    # where x is the unknown pixel values in the blended region, and b is the known pixel values from the source image and 
    # the target image at the boundary of the region.

    # ===== create Poisson matrix for b =========
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3],num_layer] # background
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3],num_layer] # generated image
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b, where x is the unknown pixel values in the blended region, and b is the known pixel values from the source image 
        # and the target image at the boundary of the region.
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>255] = 255
        x[x<0] = 0
        x = np.array(x, img_target.dtype)
        img_target[region_target[0]:region_target[2],region_target[1]:region_target[3],num_layer] = x

    return img_target
