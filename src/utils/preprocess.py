import os

import numpy as np
import pandas as pd
import torch

from PIL import Image
import cv2

def seed_setting(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
        

def get_file_path(dataset_dir, image_id):
    if os.path.exists(f"{dataset_dir}/{image_id}_thumbnail.png"):
        return f"{dataset_dir}/{image_id}_thumbnail.png"
    else:
        return f"{dataset_dir}/{image_id}.png"

def get_crop_file_path(dataset_dir, image_id, crop_id):
    if os.path.exists(f"{dataset_dir}/{image_id}_thumbnail_{crop_id}.png"):
        return f"{dataset_dir}/{image_id}_thumbnail_{crop_id}.png"
    else:
        return f"{dataset_dir}/{image_id}_{crop_id}.png"
    
def get_cropped_images(file_path, image_id, classname, config, th_area = 1000):

    image = Image.open(file_path)
    # Aspect ratio
    as_ratio = image.size[0] / image.size[1]
    
    sxs, exs, sys, eys = [],[],[],[]
    if as_ratio >= 1.5:
        # Crop
        mask = np.max( np.array(image) > 0, axis=-1 ).astype(np.uint8)
        retval, labels = cv2.connectedComponents(mask)
        if retval >= as_ratio:
            x, y = np.meshgrid( np.arange(image.size[0]), np.arange(image.size[1]) )
            for label in range(1, retval):
                area = np.sum(labels == label)
                if area < th_area:
                    continue
                xs, ys= x[ labels == label ], y[ labels == label ]
                sx, ex = np.min(xs), np.max(xs)
                cx = (sx + ex) // 2
                crop_size = image.size[1]
                sx = max(0, cx-crop_size//2)
                ex = min(sx + crop_size - 1, image.size[0]-1)
                sx = ex - crop_size + 1
                sy, ey = 0, image.size[1]-1
                sxs.append(sx)
                exs.append(ex)
                sys.append(sy)
                eys.append(ey)
        else:
            crop_size = image.size[1]
            for i in range(int(as_ratio)):
                sxs.append( i * crop_size )
                exs.append( (i+1) * crop_size - 1 )
                sys.append( 0 )
                eys.append( crop_size - 1 )
    else:
        # Not Crop (entire image)
        sxs, exs, sys, eys = [0,],[image.size[0]-1],[0,],[image.size[1]-1]

    df_crop = pd.DataFrame()
    df_crop["image_id"] = [image_id] * len(sxs)
    df_crop["file_path"] = [file_path] * len(sxs)
    df_crop["sx"] = sxs
    df_crop["ex"] = exs
    df_crop["sy"] = sys
    df_crop["ey"] = eys
    df_crop["class"] = classname
    df_crop["label"] = config.class2label.index(classname)
    
    return df_crop