from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import cv2
from PIL import Image
from glob import glob
import pdb

class Trans10kDataset(Dataset):
    def __init__(self, config, train, transform=None, colorjitter=None):
        self.root_dir = config['dataset']
        self.config = config
        if train:
            self.root_dir = os.path.join(self.root_dir, 'train')
        else:
            self.root_dir = os.path.join(self.root_dir, 'validation')
        self.imgs = sorted(glob(os.path.join(self.root_dir, "images", "*.jpg")))
        self.transform = transform
        self.img_cnt = len(self.imgs)
        assert(self.img_cnt > 0)
        self.colorjitter = colorjitter

    def __len__(self):
        return self.img_cnt

    def __getitem__(self, idx):
        rgb_name = self.imgs[idx]
        mask_name = rgb_name.replace('images', 'masks').replace('jpg', 'png')
        image = cv2.imread(rgb_name)
        image = cv2.resize(image, dsize=(self.config["img_size"]))
        mask = cv2.imread(mask_name).squeeze()
        mask = cv2.resize(mask, dsize=(self.config["img_size"]))
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)
        mask = np.transpose(mask, (2,0,1))
        image = np.transpose((image[:, :, :3]), (2, 0, 1)) / 255
        image = torch.from_numpy(image.astype(np.float32))
        mask = np.array(mask / 255, dtype=np.float32)
        mask = torch.from_numpy(mask.squeeze())
        if self.transform:
            edge = torch.stack([edge] * 3, dim=0)
            edge_image = torch.cat((image, edge), dim=0)
            edge_image = self.transform(edge_image)
            image = edge_image[:3]
            image = self.colorjitter(image)
            edge = edge_image[4]
        sample = {'rgb' : image, 'edge' : edge}
        return sample
