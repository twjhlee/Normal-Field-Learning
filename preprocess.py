"""code to predict normals and preprocess
Requires images and transforms_train.json
"""
import os
import torch
import argparse
import cv2
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="data/dex_nerf_example/images")
# for surface normal estimation
parser.add_argument("--sn_checkpoint", default='surface_normal_uncertainty/experiments/nomask_noset1/models/checkpoint_iter_0000200000.pt')
parser.add_argument("--num_augmentation", default=5)
parser.add_argument("--batch_size", default=8, type=int, help="batch size for surface normal ")
parser.add_argument('--sn_outdir', default='uncertainty')
parser.add_argument('--img_height', default=480)
parser.add_argument('--img_width', default=640)
parser.add_argument('--use_tta', action='store_true')
parser.add_argument('--dvgo_config', default='norm_kappa_ber_fixate')
parser.add_argument('--sigma_threshold', default=0.3)
args = parser.parse_args()

if __name__ == "__main__":
    # process robot end effector pose
    cmd = "python tools/ee2cam.py --input_txt {}".format(os.path.join(args.input_dir, "cam_transforms.txt"))
    os.system(cmd)

    # run surface normal estimation
    if not os.path.exists(os.path.join(args.input_dir, "uncertainty")):
        if not args.use_tta:
            os.system("python surface_normal_uncertainty/test.py \
                --checkpoint {} --imgs_dir {} --outdir {} --batch_size {}".format(
                args.sn_checkpoint,
                args.input_dir,
                args.sn_outdir,
                args.batch_size
            ))
        else:
            os.system("python surface_normal_uncertainty/test_with_uncertainty.py \
            --checkpoint {} --imgs_dir {} --outdir {} --num_augmentation {}".format(
            args.sn_checkpoint,
            args.input_dir,
            args.sn_outdir,
            args.num_augmentation
        ))

    # copy results to input dir
    os.system("cp {} {}".format(
        os.path.join(args.input_dir, args.sn_outdir, "*color_mu.png"),
        args.input_dir
    ))
    os.system("cp {} {}".format(
        os.path.join(args.input_dir, args.sn_outdir, "*.npy"),
        args.input_dir
    ))

    # change to world frame normals
    os.system("python tools/cam_normal_to_gnd.py --basedir {} --is_inmc".format(args.input_dir))

    # predict masks
    if not os.path.exists(os.path.join(args.input_dir, "uncertainty_mask")):
        os.system("python deeplab/eval.py --imgs_dir {} --ckpt_path {}".format(
            args.input_dir,
            "deeplab/checkpoints/ckpts_mask_newsplit0103.pt"
        ))
    os.system("python tools/mask_inmc.py --basedir {}".format(args.input_dir))
       