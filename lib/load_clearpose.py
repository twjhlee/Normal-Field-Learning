import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
import random
import scipy.io
from glob import glob
import matplotlib.pyplot as plt


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


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


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_clearpose_data(
    basedir,
    imgdir='.',
    half_res=False,
    testskip=1,
    datatype='rgb',
    use_depthmask=False,
    use_kappa=False,
    bg_color=None,
    mask_thresh=0.2,
    use_mask=False,
    use_bernoulli=False
):
    # Read from metadata file

    # Mask_types = [0: no mask, 1: gt mask, 2:pred mask no bernoulli, 3: pred mask bernoulli]
    # List all variants here
    if datatype == "rgb":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_masked.png")))
        masktype = 0
    elif datatype == "rgb_gt_masked":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_masked.png")))
        masktype = 1
    elif datatype == "rgb_pred_masked":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_masked.png")))
        masktype = 2
    elif datatype == "rgb_pred_masked_ber":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_masked.png")))
        masktype = 3
    elif datatype == "gt_norm":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*normal_true_gnd.png")))
        masktype = 0
    elif datatype == "gt_norm_gt_masked":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*normal_true_gnd.png")))
        masktype = 1
    elif datatype == "gt_norm_pred_masked":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*normal_true_gnd.png")))
        masktype = 2
    elif datatype == "gt_norm_pred_masked_ber":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*normal_true_gnd.png")))
        masktype = 3
    elif datatype == "norm":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*mu_gnd_pred_masked.png")))
        masktype = 0
    elif datatype == "norm_gt_masked":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*mu_gnd_pred_masked.png")))
        masktype = 1
    elif datatype == "norm_pred_masked":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*mu_gnd_pred_masked.png")))
        masktype = 2
    elif datatype == "norm_pred_masked_ber":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*mu_gnd_pred_masked.png")))
        masktype = 3
    elif datatype == "gt_mask":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*mu_gnd_pred_masked.png")))
        masktype = 1
    elif datatype == "pred_mask":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_masked.png")))
        masktype = 2
    elif datatype == "pred_mask_ber":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_masked.png")))
        masktype = 3
 
    cam_transforms = np.load(os.path.join(basedir, imgdir, 'cam_transforms.npy'))
    cam_transforms = cam_transforms.astype(np.float32)
    # Get images and poses
    H = 480
    W = 640
    f = 601

    K = np.array([
        [601, 0.0, 334],
        [0.0, 601, 248],
        [0.0, 0.0, 1.0]
    ])

    img_list = []
    pose_list = []
    for idx, i_path in enumerate(all_imgs):
        assert os.path.exists(i_path)
        img = cv2.imread(i_path, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = img.astype(np.float64) / 255

        # if input is only mask
        if datatype in ["gt_mask", "pred_mask", "pred_mask_ber"]:
            img = np.ones_like(img)
            obj_color = 1 - bg_color
            img[..., 0] *= obj_color[0]
            img[..., 1] *= obj_color[1]
            img[..., 2] *= obj_color[2]

        # no mask
        last_name = i_path.split('/')[-1]
        last_name = last_name.split('-')[-1]
        if masktype == 0:
            p_hat = np.zeros((H, W))
        # GT mask
        elif masktype == 1:
            mask = cv2.imread(i_path.replace(last_name, "label.png"), -1)
            img[mask == 0] = np.array(bg_color + [0.0])
            p_hat = np.zeros((H, W))
        # pred mask no bernoulli
        elif masktype == 2:
            img[img[..., -1] <= mask_thresh] = np.array(bg_color + [0.0])
            p_hat = np.zeros((H, W))
        # pred mask bernoulli
        elif masktype == 3:
            p_hat = np.load(i_path.replace(last_name, "p_hat.npy"))

        # TODO: kappa(test augmented) - only for normals
        if use_kappa:
            kappa_name = i_path.replace(last_name, "color_kappa.npy")
            kappa = np.load(kappa_name)
            if kappa.ndim == 3:
                kappa = kappa[:, :, 0]
            kappa = np.expand_dims(kappa, axis=-1)
        else:
            kappa = np.zeros((H, W, 1))
        img = np.concatenate([img, kappa], axis=-1)
        img_list.append(img)

        cam_pose = cam_transforms[idx]
        h_, w_ = cam_pose.shape
        if h_ == 3:
            last_row = np.array([0, 0, 0, 1]).reshape(1, 4)
            cam_pose = np.append(cam_pose, last_row, axis=0)
        cam_pose = np.matmul(cam_pose, flip_mat)
        # cam_pose = np.matmul(cam_pose, fix_pose_mat)
        cam_pose = cam_pose.astype(np.float32)
        pose_list.append(cam_pose)

    imgs = np.array(img_list)
    top_view = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1],
    ])
    pose_list[1] = top_view.astype(np.float32)

    # normalize pose
    poses = np.array(pose_list)
    # # poses[:, :3, :3] = np.linalg.inv
    # poses[:, :3, 3] -= poses[:, :3, 3].mean(axis=0)
    # poses[:, :3, 3] /= np.linalg.norm(poses[:, :3, 3], axis=1).mean()

    i_split = []
    i_split.append(np.arange(0, imgs.shape[0], 2))
    i_split.append(np.arange(1, imgs.shape[0], 2))
    i_split.append(np.arange(1, imgs.shape[0], 2))
    # IMAGE_FROM, IMAGE_TO = 134, 178
    # same train and test
    # i_split.append(np.arange(imgs.shape[0]))
    # i_split.append(np.arange(imgs.shape[0]))
    # i_split.append(np.arange(imgs.shape[0]))

    render_poses = torch.stack([pose_spherical(angle, -15, 1.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    return imgs, poses, render_poses, [H, W, f], K, i_split
