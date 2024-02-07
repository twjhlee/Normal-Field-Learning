import glob
import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class TestLoader(object):
    def __init__(self, args, fldr_path):
        self.testing_samples = CustomLoadPreprocess(args, fldr_path)
        self.data = DataLoader(self.testing_samples, args.batch_size,
                               shuffle=False,
                               # TODO change num_workers to args.num_augmentation
                               num_workers=args.batch_size,
                               pin_memory=False)



class CustomLoadPreprocess(Dataset):
    def __init__(self, args, fldr_path):
        self.fldr_path = fldr_path
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.filenames = sorted(glob.glob(os.path.join(self.fldr_path, '*-color.png')))
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB").resize(size=(self.input_width, self.input_height), resample=Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)
        img_name = (img_path.split('/'))[-1].replace('.png', '').replace('.jpg', '')

        sample = {'img': img.squeeze(),
                  'img_name': img_name}

        return sample
