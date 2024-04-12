from __future__ import print_function
import torch
import os
import cv2
import numpy as np
from models.model import Encoder, Decoder
from torch.utils.data import DataLoader
from dataset.dataset import EvalDataset
from scipy.spatial.transform import Rotation as R
import scipy.io
from tqdm import tqdm
import argparse
import json
from glob import glob
# FOV = 1.3
# META = scipy.io.loadmat("/home/twjhlee/Research/prim3d/code/nerf-pytorch/my_utils/my_data/clearpose/normal/metadata.mat")

def normal2CGnormal(normal):
    """Converts general normal images to cleargrasp type normals. Input ouput all np.arrays"""
    normal = (normal / 255).astype(np.float32)
    normal_cg = normal + 1
    normal_cg *= 127.5
    return normal_cg

def normalize(array):
    """Normalie array to be within  0 ~ 255
    """
    array = array.astype(np.float32)
    array = (array - np.min(array)) / (np.max(array) - np.min(array))
    array = array * 255
    array = array.astype(np.uint8)
    return array

def save_output(rgb, target, imgs_dir, pose=None, key=None):
    """save output to disk. rgb normals all cpu tensors"""
    rgb = rgb.numpy()
    B, C, H ,W = rgb.shape
    mask = target.numpy()
    for _idx in range(len(rgb)):
        maskname = os.path.join(imgs_dir, "{}-pred_mask.png".format(key[_idx]))
        mask_element = np.transpose(mask[_idx].reshape(1, H, W), (1, 2, 0))
        vis_mask = np.clip(mask_element, 0, 1)
        vis_mask = np.stack([vis_mask.squeeze()] * 3, axis=-1)
        vis_mask *= 255
        vis_mask = vis_mask.astype(np.uint8)
        vis_mask = cv2.resize(vis_mask, (640, 480))
        cv2.imwrite(maskname, vis_mask)


@torch.no_grad()
def evaluate(args, ckpt_path, device):
    Enc = Encoder().to(device)
    Dec = Decoder().to(device)

    ckpt = torch.load(ckpt_path)
    Enc.load_state_dict(ckpt['Enc'])
    Dec.load_state_dict(ckpt['Dec'])

    Enc.eval()
    Dec.eval()

    testset = EvalDataset(args)
    test_loader = DataLoader(
        testset,
        batch_size=4,
        num_workers=4,
        shuffle=False
    )

    for idx, data in tqdm(enumerate(test_loader)):
        rgb = data['rgb'].to(device).squeeze().contiguous()
        if rgb.ndim == 3:
            rgb = torch.unsqueeze(rgb, dim=0)
        z, ll_feat = Enc(rgb)
        target_pred = Dec(z, ll_feat, rgb.size()[2:])
        save_output(rgb.cpu(), target_pred.cpu(), args.imgs_dir, key=data['key'])

if __name__ == "__main__":
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument("--imgs_dir", default='.')
        parser.add_argument("--ckpt_path", default='.')
        args = parser.parse_args()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = args.ckpt_path
        evaluate(args, ckpt_path, device)
    
