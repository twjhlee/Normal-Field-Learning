import os
import cv2
import numpy as np
import OpenEXR
import Imath
from glob import glob
import array
import  matplotlib.pyplot as plt
import argparse

FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

parser = argparse.ArgumentParser()
parser.add_argument("--filename", required=True)
parser.add_argument("--height", default=480)
parser.add_argument("--width", default=640)
parser.add_argument("--norm2color", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    gt_name = args.filename
    file = OpenEXR.InputFile(gt_name)
    (R,G,B,A) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B", "A") ]
    R = np.array(R).reshape(int(args.height), int(args.width))
    G = np.array(G).reshape(int(args.height), int(args.width))
    B = np.array(B).reshape(int(args.height), int(args.width))
    A = np.array(A).reshape(int(args.height), int(args.width))
    img = np.stack([R, G, B, A], axis=-1)
    if args.norm2color:
        img = (img + 1) * 0.5
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(gt_name.replace(".exr", ".png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
