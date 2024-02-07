from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split as split
from torchvision import transforms as T
import cv2
from PIL import Image
from glob import glob
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import scipy.io


class RGB2NormalDataset(Dataset):
    def __init__(self, config, train, random_state, transform=None, colorjitter=None):
        # print('making dataset class')
        self.config = config
        all_pngs = sorted(glob(os.path.join(config['dataset_dir'], '*color.png')))
        all_imgs = all_pngs

        if len(all_imgs) == 0:
            newpath = config['dataset_dir'].replace('twjhlee', 'junholee').replace('Data_ssd', 'Data')
            all_pngs = sorted(glob(os.path.join(newpath, '*color.png')))
            all_imgs = all_pngs
        
        assert len(all_imgs) > 0

        # Train test split 
        train_img, val_img = split(all_imgs, random_state=random_state)
        if train:
            self.imgs = train_img
        else:
            self.imgs = val_img
        del train_img, val_img

        self.transform = transform
        self.colorjitter = colorjitter

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        rgb_name = self.imgs[idx]
        output_mod = self.config['output_mod']
        if "depth" == output_mod:
            target_name = rgb_name.replace('color', 'depth')
        elif "normal" == output_mod:
            target_name = rgb_name.replace('color.png', 'normal_true.png')
        elif "mask" == output_mod:
            target_name = rgb_name.replace('color', 'label')
        else:
            print("Unsupported output mod. Check train.json")

        image = cv2.imread(rgb_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.config['img_size'], self.config['img_size']))
        image = np.transpose((image[:, :, :3]), (2, 0, 1)) / 255
        image_ten = deepcopy(torch.from_numpy(image.astype(np.float32)))

        target = cv2.imread(target_name, -1)

        if target.ndim == 2:
            target = np.expand_dims(target, axis=-1)
        target = cv2.resize(target, dsize=(self.config['img_size'], self.config['img_size']))
        if "mask" in output_mod:
            target = target.reshape(self.config['img_size'], self.config['img_size'], 1)
            target[target > 0] = 1
            target = target.astype(np.float32)
        
        if "depth" in output_mod:
            target = target * 0.001
            target = target.astype(np.float32)
            if target.ndim == 2:
                target = np.expand_dims(target, axis=-1)
        elif "normal" in output_mod:
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
            H, W, C = target.shape

            # Get valid mask
            valid_mask = (target[:, :, 0] == 127) * (target[:, :, 2] == 127) * (target[:, :, 1] == 127)
            valid_mask = 1 - valid_mask
            valid_mask_ten = torch.from_numpy(valid_mask.astype(np.float32)).reshape(1, H, W)
            # debug - save target_vis
            target = target.astype(np.float32) / 127.5
            target = target - 1
            # target = target - 1
            # norm = target[:, :, 0] ** 2 + target[:, :, 1] ** 2 + target[:, :, 2] ** 2
            # norm = np.sqrt(norm)
            # target[norm > 0.9, 2] /= norm[norm > 0.9]

            # # get spherical coordinates
            # theta = np.arccos(target[:, :, 2])
            # phi = np.arctan2(target[:, :, 1], target[:, :, 0])

            # # debug - visualize theta and phi
            # theta_vis = theta * 255 / np.pi
            # theta_vis = theta_vis.astype(np.uint8)
            # theta_vis = np.stack([theta_vis] * 3, axis=-1)
            # # theta_vis = cv2.applyColorMap(theta_vis, cv2.COLORMAP_TURBO)

            # phi_vis = phi + np.pi
            # phi_vis = phi_vis * 255 / (2 * np.pi)
            # phi_vis = phi_vis.astype(np.uint8)
            # phi_vis = np.stack([phi_vis] * 3, axis=-1)
            # # phi_vis = cv2.applyColorMap(phi_vis, cv2.COLORMAP_TURBO)
            # total_vis = np.concatenate([target_vis, theta_vis, phi_vis], axis=1)
            # plt.imshow(total_vis)
            # plt.show()

        target = np.transpose(target, (2, 0, 1))
        target_ten = deepcopy(torch.from_numpy(target.astype(np.float32)))
        
        if self.transform:
            if "normal" == output_mod:
                input_ten = torch.cat((image_ten, target_ten, valid_mask_ten), dim=0)
            else:
                input_ten = torch.cat((image_ten, target_ten), dim=0)
            input_ten = self.transform(input_ten)
            image_ten = deepcopy(input_ten[:3])
            image_ten = self.colorjitter(image_ten)
            target_ten = input_ten[3:]
        sample = {'rgb' : deepcopy(image_ten), 'target' : deepcopy(target_ten), }

        return sample


class EvalDataset(Dataset):
    def __init__(self, config):
        self.config = config
        all_imgs = sorted(glob(os.path.join(config.imgs_dir, "*color.png")))
        # self.META = scipy.io.loadmat("/home/twjhlee/Research/prim3d/code/nerf-pytorch/my_utils/my_data/clearpose/normal/metadata.mat")
        assert len(all_imgs) > 0
        self.imgs = all_imgs

    def __len__(self):
        return(len(self.imgs))

    def __getitem__(self, idx):
        fname = self.imgs[idx]

        key = fname.split('/')[-1].replace("-color.png", "").replace("_rgb.png", "")

        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, C = img.shape
        # center_h = H // 2
        # height = [center_h - W // 2, center_h + W // 2]
        # img = img[height[0]: height[1]]
        img = cv2.resize(img, dsize=(480, 480))
        img = np.transpose((img[:, :, :3]), (2, 0, 1)) / 255
        img_ten = deepcopy(torch.from_numpy(img.astype(np.float32)))
        
        sample = {'rgb': img_ten, "key": key}
        return sample

class TTADataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.num_aug = config.num_aug
        all_imgs = sorted(glob(os.path.join(config.imgs_dir, "*color.png")))
        assert len(all_imgs) > 0
        self.imgs = all_imgs

    def __len__(self):
        return(len(self.imgs))

    def __getitem__(self, idx):
        fname = self.imgs[idx]

        key = fname.split('/')[-1].replace("-color.png", "").replace("_rgb.png", "")

        img = Image.open(fname).convert("RGB").resize(size=(480, 480), resample=Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        H, W, C = img.shape
        imgs_aug = []
        for _idx in range(int(self.num_aug)):
            if _idx > 0:
                # img_aug = T.functional.adjust_hue(img, (1 / self.num_aug) * ((_idx + 1) // 2) * (-1 ** _idx))
                img_aug = img
                img_aug = T.functional.adjust_hue(img, ((1 / self.num_aug) * ((_idx + 1) // 2)) * (-1 ** _idx))
                # img_aug = T.functional.adjust_contrast(img_aug, 1 * _idx / self.num_aug + 0.5)
                # img_aug = T.functional.adjust_sharpness(img_aug, 1 * _idx / self.num_aug + 0.5)
                # img_aug = T.functional.adjust_gamma(img_aug, 1 * _idx / self.num_aug + 0.5)

            else:
                img_aug = img
            imgs_aug.append(img_aug)

        sample = {'rgb': torch.stack(imgs_aug, dim=0), "key": key}
        return sample