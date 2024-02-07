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
from multiprocessing import Process


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
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w

def load_inmc_data(
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
        use_bernoulli=False,
        train_step=2
    ):
    # Read from metadata file

    if datatype == "rgb":
        # deprecated
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color.png")))
    elif datatype == "rgb_pred_masked":
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_masked.png")))
    elif datatype == 'pred_normal':
        # deprecated
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*color_mu_gnd.png")))
    elif datatype == 'pred_normal_pred_masked' or datatype == 'pred_mask':
        all_imgs = sorted(glob(os.path.join(basedir, imgdir, "*mu_gnd_pred_masked.png")))
    cam_transforms = np.load(os.path.join(basedir, imgdir, 'cam_transforms.npy'))
    cam_transforms = cam_transforms.astype(np.float32)
    # Get images and poses
    H = 480
    W = 640
    f = 0

    # for realsense
    K = np.array([
        [612.33475388, 0.0, 318.91296108],
        [0.0, 612.30676159, 241.45904916],
        [0.0, 0.0, 1.0]
    ])

    if "Blender" in basedir:
        K = None
        H = 480
        W = 640
        f = 812.3673490683944

    if "dex_nerf" in basedir or "dexnerf" in basedir:
        K = None
        H = 800
        W = 1236
        f = 961.0007468423943


    # K = np.array([
    #     [f, 0.0, 320.071],
    #     [0.0, f, 243.705],
    #     [0.0, 0.0, 1.0]
    # ])

    img_list = []
    pose_list = []
    for idx, i_path in enumerate(all_imgs):
        assert os.path.exists(i_path)
        img = cv2.imread(i_path, -1)
        img = cv2.resize(img, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = img.astype(np.float32) / 255
        if datatype == "pred_mask":
            img[..., 0] = 1
            img[..., 1] = 1
            img[..., 2] = 1
            img = img.astype(np.float64)
       
        # TODO: mask sampling
        if use_mask:
            if use_bernoulli:
                p_hat = cv2.imread(i_path, -1)
                p_hat = cv2.resize(p_hat, (W, H))
                p_hat = p_hat[..., -1].astype(np.float32)
                p_hat = p_hat / 255
                # Maybe add some value(offset)
                p_hat += 0.1
                p_hat = np.clip(p_hat, 0, 1)
                p_hat = cv2.resize(p_hat, (W, H))
            else:
                assert img.shape[-1] == 4
                img[img[..., -1] <= mask_thresh] = np.array(bg_color + [0.0])
                p_hat = np.zeros((H, W))
        else:
            p_hat = np.zeros((H, W))
        img = img[..., :3]
        img = np.concatenate([img, p_hat[..., None]], axis=-1)

        # TODO: kappa(test augmented) - only for normals
        if use_kappa:
            kappa_name = i_path.replace("mu_gnd_pred_masked.png", "color_kappa.npy")
            kappa = np.load(kappa_name)
            kappa = cv2.resize(kappa, (W, H))
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

        if "INMC" in basedir or "inmc" in basedir:
            cam_pose = np.matmul(cam_pose, flip_mat)
            cam_pose = np.matmul(cam_pose, fix_pose_mat)
        elif "clearpose" in basedir or "ClearPose" in basedir:
            cam_pose = np.matmul(cam_pose, flip_mat)
        cam_pose = cam_pose.astype(np.float32)
        pose_list.append(cam_pose)

    imgs = np.array(img_list)
    top_view = np.array([
        [1, 0, 0, 0.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1],
    ])
    ###### if we need to add top view
    #pose_list[1] = top_view.astype(np.float32)
    
    # normalize pose
    poses = np.array(pose_list)
    # # poses[:, :3, :3] = np.linalg.inv
    # poses[:, :3, 3] -= poses[:, :3, 3].mean(axis=0)
    # poses[:, :3, 3] /= np.linalg.norm(poses[:, :3, 3], axis=1).mean()


    ## if we need to see camera poses
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(poses[:, 0, 0], poses[:, 0, 1], poses[:, 0, 2])
    # plt.show()

    i_split = []
    i_split.append(np.arange(0, imgs.shape[0], train_step))
    if "Blender" in basedir:
        i_split.append(np.arange(0, imgs.shape[0], 10))
        i_split.append(np.arange(1, imgs.shape[0], 10))
    else:
        i_split.append(np.arange(1, imgs.shape[0], train_step))
        i_split.append(np.arange(1, imgs.shape[0], train_step * 2))
    # IMAGE_FROM, IMAGE_TO = 134, 178
    # same train and test
    # i_split.append(np.arange(imgs.shape[0]))
    # i_split.append(np.arange(imgs.shape[0]))
    # i_split.append(np.arange(imgs.shape[0]))
        
    render_poses = torch.stack([pose_spherical(angle, -15, 1.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    return imgs, poses, render_poses, [H, W, f], K, i_split