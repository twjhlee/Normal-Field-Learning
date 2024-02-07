import os
from glob import glob
from tqdm import tqdm

PATH = "/home/twjhlee/Data/RGB2Normal/clearpose_new/train/*/*/*color.png"
all_imgs = glob(PATH)
for c_name in tqdm(all_imgs):
    exist_normal = os.path.exists(c_name.replace('color.png', 'normal_true.png'))
    exist_label = os.path.exists(c_name.replace('color.png', 'label.png'))
    both = exist_normal * exist_label
    if not both:
        print(c_name)
        print(exist_normal)
        print(exist_label)

