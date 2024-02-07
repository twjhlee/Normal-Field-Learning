import os
from glob import glob
from tqdm import tqdm

all_colors = glob("/home/twjhlee/Data/ClearPose_orig/*/*/*color.png")
for c_name in tqdm(all_colors):
    n_name = c_name.replace("color.png", "normal_true.png")
    exists = os.path.exists(n_name)
    if not exists:
        os.remove(c_name)
