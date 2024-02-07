import os
import cv2
import numpy as np
from glob import glob
import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import data.utils as data_utils


RANDOM = 41

def get_loaders(args):
    all_rgbs = glob(os.path.join(args.dataset_dir, "*/*/*/*-color.png"))
    train_rgb, val_rgb = train_test_split(all_rgbs, test_size=0.01, random_state=RANDOM)
    train_proc = FinetunePreprocess(args, "train", train_rgb)
    val_proc = FinetunePreprocess(args, "test", val_rgb)
    train_loader = DataLoader(train_proc, args.batch_size,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True)
    val_loader = DataLoader(val_proc, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False)
    
    return train_loader, val_loader


class FinetuneLoader(object):
    def __init__(self, args, mode, rgb_list):
        """mode: {'train_big',  # training set used by GeoNet (CVPR18, 30907 images)
                  'train',      # official train set (795 images) 
                  'test'}       # official test set (654 images)
        """
        self.t_samples = FinetunePreprocess(args, mode, rgb_list)

        # train, train_big
        if 'train' in mode:
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.t_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.t_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   drop_last=True,
                                   sampler=self.train_sampler)

        else:
            self.data = DataLoader(self.t_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False)


class FinetunePreprocess(Dataset):
    def __init__(self, args, mode, rgb_list):
        self.args = args
        # train, train_big, test, test_new

        self.all_rgbs = rgb_list
        self.mode = mode
        # normalize on img?
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.input_width = args.input_width
        self.input_height = args.input_height

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        # img path and norm path
        img_path = self.all_rgbs[idx]
        norm_path = img_path.replace("color.png", "normal_true.png")
        mask_path = img_path.replace('color.png', "label.png")
        scene_name = self.mode
        img_name = img_path.split('/')[-1].split('.png')[0]

        # read img / normal
        img = Image.open(img_path).convert("RGB").resize(size=(self.input_width, self.input_height), 
                                                            resample=Image.BILINEAR)
        norm_gt = Image.open(norm_path).convert("RGB").resize(size=(self.input_width, self.input_height), 
                                                            resample=Image.NEAREST)
        mask = Image.open(mask_path).convert("RGB").resize(size=(self.input_width, self.input_height), 
                                                            resample=Image.NEAREST)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask[mask > 0] = 1

        if 'train' in self.mode:
            # horizontal flip (default: True)
            DA_hflip = False
            if self.args.data_augmentation_hflip:
                DA_hflip = random.random() > 0.5
                if DA_hflip:
                    img = TF.hflip(img)
                    norm_gt = TF.hflip(norm_gt)

            # to array
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)

            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 127, norm_gt[:, :, 1] == 127),
                    norm_gt[:, :, 2] == 127))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]
            # norm_valid_mask = (np.stack([mask] * 3, axis=-1)).astype(np.bool_)
            # plt.imshow((norm_valid_mask*255).astype(np.uint8))
            # plt.savefig("mask_ex.png")

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
            norm = np.linalg.norm(norm_gt, axis=-1)
            norm = np.stack([norm] * 3, axis=-1)
            norm_gt = norm_gt / norm

            if DA_hflip:
                norm_gt[:, :, 0] = - norm_gt[:, :, 0]

            # random crop (default: False)
            if self.args.data_augmentation_random_crop:
                img, norm_gt, norm_valid_mask = data_utils.random_crop(img, norm_gt, norm_valid_mask, 
                                                                     height=self.input_height, width=self.input_width)

            # color augmentation (default: True)
            if self.args.data_augmentation_color:
                if random.random() > 0.5:
                    img = data_utils.color_augmentation(img, indoors=True)
            
            breakpoint()
        else:
            img = np.array(img).astype(np.float32) / 255.0

            norm_gt = np.array(norm_gt).astype(np.uint8)
            norm_valid_mask = (np.stack([mask] * 3, axis=-1)).astype(np.bool_)
            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 127, norm_gt[:, :, 1] == 127),
                    norm_gt[:, :, 2] == 127))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
            norm = np.linalg.norm(norm_gt, axis=-1)
            norm = np.stack([norm] * 3, axis=-1)
            norm_gt = norm_gt / norm

        # to tensors
        img = self.normalize(torch.from_numpy(img).permute(2, 0, 1))            # (3, H, W)
        norm_gt = torch.from_numpy(norm_gt).permute(2, 0, 1)                    # (3, H, W)
        norm_valid_mask = torch.from_numpy(norm_valid_mask).permute(2, 0, 1)    # (1, H, W)

        sample = {'img': img,
                  'norm': norm_gt,
                  'norm_valid_mask': norm_valid_mask,
                  'scene_name': scene_name,
                  'img_name': img_name}

        return sample
