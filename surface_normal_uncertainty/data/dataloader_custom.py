import glob
import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class CustomLoader(object):
    def __init__(self, args, fldr_path):
        self.testing_samples = CustomLoadPreprocess(args, fldr_path)
        self.data = DataLoader(self.testing_samples, 1,
                               shuffle=False,
                               # TODO change num_workers to args.num_augmentation
                               num_workers=0,
                               pin_memory=False)



class CustomLoadPreprocess(Dataset):
    def __init__(self, args, fldr_path):
        self.num_aug = args.num_augmentation
        self.fldr_path = fldr_path
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.filenames = sorted(glob.glob(os.path.join(self.fldr_path, '*color.png')))
        self.input_height = args.input_height
        self.input_width = args.input_width
        self.augmentation = T.ColorJitter(brightness=0.5, hue=0.3)
        # self.augmentation = T.RandomVerticalFlip(0.5)
        # self.augmentation = T.RandomHorizontalFlip(0.5)
        # self.augmentation = T.RandomResizedCrop((480, 640), scale=(0.5, 1.0))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB").resize(size=(self.input_width, self.input_height), resample=Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        imgs_aug = []
        for _idx in range(self.num_aug):
            if _idx > 0:
                img_aug = T.functional.adjust_hue(img, ((1 / self.num_aug) * ((_idx + 1) // 2)) * (-1 ** _idx))
                # img_aug = T.functional.adjust_contrast(img_aug, 1 / self.num_aug * _idx + 0.5)
                # img_aug = T.functional.adjust_sharpness(img_aug, 1 * _idx / self.num_aug + 0.5)
                # img_aug = T.functional.adjust_gamma(img_aug, 1 * _idx / self.num_aug + 0.5)
            else:
                img_aug = img
            img_aug = self.normalize(img_aug)
            imgs_aug.append(img_aug)

        img_name = img_path.split('/')[-1]
        img_name = img_name.split('.png')[0] if '.png' in img_name else img_name.split('.jpg')[0]

        sample = {'img': torch.stack(imgs_aug, dim=0).squeeze(),
                  'img_name': img_name}

        return sample
