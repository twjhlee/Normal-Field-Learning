import numpy as np
import os
import argparse 


EE2CAM = [0.055, 0.0325, -0.05] #numbers from cad
parser = argparse.ArgumentParser()
parser.add_argument("--input_txt", required=True)
args = parser.parse_args()

if __name__ == "__main__":
    # Read ee pose from txt file
    assert os.path.exists(args.input_txt)
    ee_pose = np.loadtxt(args.input_txt)
    ee_pose = np.transpose(ee_pose.reshape(len(ee_pose), 4, 4), (0, 2, 1))

    # Change to cam pose
    cam_pose = ee_pose.copy()
    cam_pose[:, :3, 3] = ee_pose[:, :3, 3] + np.matmul(ee_pose[:, :3, :3], np.array(EE2CAM))

    # Save to disk
    np.save(args.input_txt.replace('.txt', '.npy'), cam_pose)
    print("Saved {}".format(args.input_txt.replace(".txt", ".npy")))
    