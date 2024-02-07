import os
from glob import glob
from tqdm import tqdm

all_rgbs = sorted(glob("/home/twjhlee/Data/ClearPose_orig/*/*/*color.png"))
print("Found {} in total".format(len(all_rgbs)))


os.makedirs("/home/twjhlee/Data_ssd/RGB2Normal/clearpose_new/train/color", exist_ok=True)
os.makedirs("/home/twjhlee/Data_ssd/RGB2Normal/clearpose_new/train/depth", exist_ok=True)
os.makedirs("/home/twjhlee/Data_ssd/RGB2Normal/clearpose_new/train/depth_true", exist_ok=True)
os.makedirs("/home/twjhlee/Data_ssd/RGB2Normal/clearpose_new/train/mask", exist_ok=True)
os.makedirs("/home/twjhlee/Data_ssd/RGB2Normal/clearpose_new/train/normal", exist_ok=True)
os.makedirs("/home/twjhlee/Data_ssd/RGB2Normal/clearpose_new/train/normal_gnd", exist_ok=True)
for cnt in tqdm(range(len(all_rgbs))):
    rgb_name = all_rgbs[cnt]
    depth_name = rgb_name.replace('color', 'depth')
    depth_true_name = rgb_name.replace('color', 'depth_true')
    mask_name = rgb_name.replace('color', 'label')
    normal_name = rgb_name.replace('color', 'normal_true')
    normal_gnd_name = rgb_name.replace('color', 'normal_gnd')

    new_rgb_name = "/home/twjhlee/Data_ssd/RGB2Normal/clearpose_new/train/color/{:07d}-color.png".format(cnt)
    cmd = "cp {} {}".format(rgb_name, new_rgb_name)
    os.system(cmd)

    new_name = new_rgb_name.replace('color', 'depth')
    cmd = "cp {} {}".format(depth_name, new_name)
    os.system(cmd)

    new_name = new_rgb_name.replace('color', 'depth_true')
    cmd = "cp {} {}".format(depth_true_name, new_name)
    os.system(cmd)

    new_name = new_rgb_name.replace('color', 'mask')
    cmd = "cp {} {}".format(mask_name, new_name)
    os.system(cmd)

    new_name = new_rgb_name.replace('color', 'normal')
    cmd = "cp {} {}".format(normal_name, new_name)
    os.system(cmd)

    new_name = new_rgb_name.replace('color', 'normal_gnd')
    cmd = "cp {} {}".format(normal_gnd_name, new_name)
    os.system(cmd)
