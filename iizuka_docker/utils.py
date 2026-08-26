import cv2
import numpy as np


# this function is used to convert a tensor to a cv2 image
def tensor2cvimg(src):
    '''return np.array
        uint8
        [0, 255]
        BGR
        (H, W, C)
    '''
    out = src.copy() * 255
    out = out.transpose((1, 2, 0)).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return out


# this function is used to convert a cv2 image to a tensor (inverse of above function)
def cvimg2tensor(src): 
    out = src.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)      # we change the color space from BGR to RGB, because the model was trained on RGB images
    out = out.transpose((2,0,1)).astype(np.float64) # we change the order of the dimensions from (H, W, C) to (C, H, W)
    out = out / 255

    return out
