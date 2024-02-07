import numpy as np
import matplotlib.pyplot as plt
import argparse
from glob import glob
import os
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)

    args = parser.parse_args()

    depth_file_paths = sorted(glob(os.path.join(args.input_dir, '*.bin')))

    for raw_depth_fname in tqdm(depth_file_paths):
        with open(raw_depth_fname, "rb") as f:
            bytes_read = f.read()

        depths = []
        for i in range(len(bytes_read))[::2]:
            depth_1000 = (bytes_read[i+1] << 8) + bytes_read[i]
            depths.append(depth_1000 / 1000.0)

        depths = np.array(depths)
        depth_img = depths.reshape((480, 640))

        np.save(raw_depth_fname.replace('.bin', '.npy'), depth_img)
        # plt.imsave(raw_depth_fname.replace('.bin', '.png'), depth_img)
        plt.imshow(depth_img)
        plt.colorbar()
        plt.clim(0, 1)
        plt.savefig(raw_depth_fname.replace('.bin', '.png'))
        plt.clf()


if __name__ == '__main__':
    main()