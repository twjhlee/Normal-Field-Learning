import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", default='Realsense_calibration/latest')
parser.add_argument("--img_height", type=int, default=480)
parser.add_argument("--img_width", type=int, default=640)
parser.add_argument("--use_tta", action='store_true')
args = parser.parse_args()

all_rgbs = sorted(glob(os.path.join(args.basedir, "*-color.png")))

for rgb_name in tqdm(all_rgbs):
    rgb = cv2.imread(rgb_name)
    mu_gnd = cv2.imread(rgb_name.replace('color.png', 'color_mu_gnd.png'), -1)
    mu_gnd = cv2.resize(mu_gnd, (args.img_width, args.img_height))
    mask = cv2.imread(rgb_name.replace('color.png', 'pred_mask.png'), -1)
    mask = cv2.resize(mask, (args.img_width, args.img_height))
    masked_rgb_name = rgb_name.replace('color.png', 'color_masked.png')
    masked_mu_name = rgb_name.replace('color.png', 'mu_gnd_pred_masked.png')

    if mask.ndim == 3:
        mask = mask[..., 0]

    masked_rgb = np.concatenate([rgb, mask[..., None]], axis=-1)
    masked_mu = np.concatenate([mu_gnd, mask[..., None]], axis=-1)

    cv2.imwrite(masked_rgb_name, masked_rgb)
    cv2.imwrite(masked_mu_name, masked_mu)