import os
import torch
import numpy as np
import imageio 
import json
from scipy.spatial.transform import Rotation as R

def load_egocentric_data(basedir):
    sub_path = os.path.join('output_dir', 'colmap', 'images.txt')
    pose_file_path = os.path.join(basedir, sub_path)

    rays2cam = np.array([
            [1., 0., 0., 0.],
            [0.,-1., 0., 0.],
            [0., 0.,-1., 0.],
            [0., 0., 0., 1.]
            ])

    world_align = np.array([
            [1., 0., 0., 0.], 
            [0., 0., 1., 0.],
            [0., -1., 0., 0.],
            [0., 0., 0., 1.]
            ])

    poses_dict = {}
    i = 0
    with open(pose_file_path) as f:
        lines = f.readlines()[4:] # discard file info
            
        for line in lines:
            tokens = line.split()
            if tokens[0] == '#':
                continue
            i += 1
            if i % 2 == 0:
                continue

            quat, t, img_fname = np.array(list(map(float, tokens[1:5]))), np.array(list(map(float, tokens[5:8]))), tokens[9]
            quat = quat[[1, 2, 3, 0]]

            rot = R.from_quat(quat).as_matrix()
            w2c = np.concatenate((rot, t[:,np.newaxis]), axis=1)
            w2c = np.concatenate((w2c, [[0,0,0,1]]), axis=0)
            c2w = np.linalg.inv(w2c)

            poses_dict[img_fname] = world_align @ c2w @ rays2cam

    '''
    normalize pose s.t. r=1, center=(0,0)
    '''
    cam_center = np.zeros(3)
    for pose in poses_dict.values():
        cam_center += pose[:3, 3]
    cam_center /= len(poses_dict)
    
    dist = 0
    for pose in poses_dict.values():
        dist += np.sqrt(((pose[:3, 3] - cam_center) ** 2).sum())

    dist /= len(poses_dict)

    for pose in poses_dict.values():
        pose[:3, 3] = (pose[:3, 3] - cam_center) / dist


    fname_train = []
    fname_test = []

    with open(os.path.join(basedir, 'train.txt')) as train_file:
        while True:
            line = train_file.readline()
            if not line:
                break
            fname_train.append(int(line.strip()))

    with open(os.path.join(basedir, 'test.txt')) as test_file:
        while True:
            line = test_file.readline()
            if not line:
                break
            fname_test.append(int(line.strip()))

    imgs = []
    poses = []

    img_dir = os.path.join(basedir, 'imgs_2')
    all_img_files = os.listdir(img_dir)
    img_files = [os.path.basename(f) for f in sorted(all_img_files, key=lambda fname: int(fname.split('.')[0])) if f.endswith('.png')]
    img_files = np.array(img_files)

    flags = np.zeros_like(img_files, dtype=bool)
    for i in range(len(img_files)):
        flags[i] = img_files[i] in poses_dict

    img_files = img_files[flags]

    i_train = []
    i_test = []
    
    for f_train in fname_train:
        i_train.append(list(img_files).index(str(f_train) + '.png'))
    for f_test in fname_test:
        i_test.append(list(img_files).index(str(f_test) + '.png'))
        
    for fname in img_files:
        imgs.append(imageio.imread(os.path.join(img_dir, fname)))
        poses.append(poses_dict[fname])

    imgs = (np.array(imgs) / 255.).astype(np.float32) 
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])
    poses = np.array(poses).astype(np.float32)
    
    H, W = imgs[0].shape[:2]

    render_poses = poses[i_test]
        
    return imgs, poses, render_poses, [H, W, None], i_train, i_test