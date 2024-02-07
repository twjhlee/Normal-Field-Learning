import os
import torch
import numpy as np
import imageio 
import json

def load_omniblender_data(basedir):
    with open(os.path.join(basedir, 'transform.json'), 'r') as fp:
        meta = json.load(fp)

    imgs = []
    poses = []
        
    for frame in meta['frames']:
        fname = os.path.join(basedir, 'images', frame['file_path'])
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32) 
    if imgs.shape[-1] == 4:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:])
    poses = np.array(poses).astype(np.float32)
    
    H, W = imgs[0].shape[:2]

    i_train = []
    i_test = []

    with open(os.path.join(basedir, 'train.txt')) as train_file:
        while True:
            line = train_file.readline()
            if not line:
                break
            i_train.append(int(line.strip()))

    with open(os.path.join(basedir, 'test.txt')) as test_file:
        while True:
            line = test_file.readline()
            if not line:
                break
            i_test.append(int(line.strip()))

    # i_train = np.arange(len(imgs))[::2]
    # i_test = np.arange(len(imgs))[1::2][:5]

    render_poses = poses[i_test]
        
    return imgs, poses, render_poses, [H, W, None], i_train, i_test


