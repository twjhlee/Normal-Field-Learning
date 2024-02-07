import os
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import scipy.io
from scipy.spatial.transform import Rotation as R
import argparse

from multiprocessing import Process

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

fix_pose_mat = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

flip_xy = np.array([
    [-1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", default="Realsense_calibration/latest")
    parser.add_argument("--is_inmc", action='store_true')
    parser.add_argument("--fname", default="color_mu.png")
    args = parser.parse_args()
    
    all_normals = sorted(glob(os.path.join(args.basedir, "*{}".format(args.fname))))
    cam_transforms = np.load(os.path.join(args.basedir, "cam_transforms.npy"))

    for idx, normal_name in tqdm(enumerate(all_normals)):
        # Read normal
        assert os.path.exists(normal_name)
        normal = cv2.imread(normal_name, -1)
        normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        normal_mask = (normal[:, :, 0] == 127) * (normal[:, :, 1] == 127) * (normal[:, :, 2] == 127)
        normal_mask = 1 - normal_mask
        normal = (normal.astype(np.float32)) / 127.5
        normal = normal - 1
        norm = np.linalg.norm(normal, axis=-1)
        norm = np.stack([norm] * 3, axis=-1)
        normal = normal / norm
        H, W, C = normal.shape

        # Debug
        normal = normal.reshape(-1, 3)
        normal = np.matmul(normal, flip_xy[:3, :3])
        normal = normal.reshape(H, W, C)
        
        # Get transform to camera
        cam_pose = cam_transforms[idx]
        # rotate was all off
        if args.is_inmc:
            cam_pose = np.matmul(cam_pose, flip_mat)
            # disable in case of clearpose
            cam_pose = np.matmul(cam_pose, fix_pose_mat)
        cam_rot = R.from_matrix(cam_pose[:3, :3])
        
        # Apply transform to image
        normal_ground = cam_rot.apply(normal.reshape(-1, 3))
        normal_ground = normal_ground.reshape(H, W, C)

        # Save altered normal to disk
        new_name = normal_name.replace(args.fname, args.fname.replace(".png", "_gnd.png"))
        norm = np.linalg.norm(normal_ground, axis=-1)
        norm = np.stack([norm] * 3, axis=-1)
        normal_ground = normal_ground / norm

        # Debug
        normal_ground = normal_ground + 1
        normal_ground *= 127.5
        # normal_ground *= 255
        # End debug
        
        
        normal_ground = np.clip(normal_ground, 0, 255)
        normal_ground = normal_ground.astype(np.uint8)
        normal_ground = cv2.cvtColor(normal_ground, cv2.COLOR_BGR2RGB)
        normal_ground[normal_mask==0, 0] = 127
        normal_ground[normal_mask==0, 1] = 127
        normal_ground[normal_mask==0, 2] = 127
        cv2.imwrite(new_name, normal_ground)