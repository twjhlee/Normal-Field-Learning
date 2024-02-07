import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
import OpenEXR
import Imath
import array

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, bg_color=[0, 0, 0], use_mask=False, train_step=2, H=480, W=640):
    splits = ['train', 'val', 'test']
    valid_splits = []
    metas = {}
    for s in splits:
        if os.path.exists(os.path.join(basedir, 'transforms_{}.json'.format(s))):
            with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
                metas[s] = json.load(fp)
            valid_splits.append(s)
    del splits
    splits = valid_splits
    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip

        for frame in meta['frames'][::skip]:
            # fname = os.path.join(basedir, frame['file_path'] + '.png')
            fname = os.path.join(basedir, frame['file_path'])

            # always read masks as well
            if '.png' in fname:
                # for rgb
                if not os.path.exists(fname):
                    fname = fname.replace('.png', '-color.png')
                img_ = cv2.imread(fname)
                img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
                img_ = img_.astype(np.float32) / 255.0
                img = img_[..., :3]#RGB or normal
                # get masks
                mask_name = fname.replace("rgb1", "gt_mask").replace('-color.png', '.png')
            elif '.exr' in fname:
                # for normals
                normal_file = OpenEXR.InputFile(fname)
                (R, G, B, A) = [array.array('f', normal_file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "A")]
                R = np.array(R)
                G = np.array(G)
                B = np.array(B)
                normal = np.stack([R, G, B], axis=-1).reshape(H, W, 3).astype(np.float32)
                img = (normal + 1) / 2
                mask_name = fname.replace("gt_normal", "gt_mask").replace('.exr', '.png')
            else:
                # Dex NeRF
                fname = fname + ".png"
                if os.path.exists(fname):
                    img_ = cv2.imread(fname)
                    img_ = cv2.cvtColor(img_, cv2.COLOR_RGB2BGR)
                    img_ = img_.astype(np.float32) / 255.0
                    img = img_[..., :3]#RGB or normal
                    mask_name = fname
                else:
                    print("Unexpected file type! Should be one of [png, exr]")
                    os._exit(0)
            
            mask_img = cv2.imread(mask_name, -1)
            # assert mask_img.shape[-1] == 4
            # assert alpha channel of mask
            mask = mask_img[..., -1] == np.max(np.unique(mask_img[..., -1]))
        
            H, W, C = img.shape
            if use_mask:
                if C == 3:
                    img[mask == 0] = np.array(bg_color)
                elif C == 4:
                    img[mask == 0] = np.array(bg_color + [0.0])
            # two dummy layers for p_hat and kappa
            dummy = np.zeros((H, W, 2))
            img = np.concatenate([img[..., :3], dummy], axis=-1)
            imgs.append(img)
            poses.append(np.array(frame['transform_matrix']))
        # imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        imgs = np.array(imgs)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    if len(metas) == 3:
        i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    else:
        i_train = np.arange(0, len(imgs), train_step)
        i_test = np.arange(0, len(imgs), 10)
        i_valid = []
        i_split = [i_train, i_valid, i_test]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    # poses[:, :3, 3] /= np.abs(poses[:, :3, 3]).max()

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
    return imgs, poses, render_poses, [H, W, focal], i_split


