import argparse
import numpy as np
import mcubes
from cluster_points import cluster_points
import os
from glob import glob
from tqdm import tqdm


def run(path, use_cluster):
    data = np.load(path)
    alpha = data['alpha']
    # alpha = alpha[..., :70]
    alpha = alpha[..., 1:]


    alpha_mask = alpha > args.thres

    xyz = np.stack(alpha_mask.nonzero(), -1) 
    print("Meshing bbox: [0.2, -0.3, 0.02, 0.8, 0.3, 0.3]")
    # xyz_min_fine = np.array([3.04, 2.25, 0.6])
    xyz_min_fine = np.array([0.2, -0.3, 0.02])
    # xyz_max_fine = np.array([4.84, 3.67, 1.5])
    xyz_max_fine = np.array([0.8, 0.3, 0.3])

    scene_size = xyz_max_fine - xyz_min_fine
    voxel_size = scene_size / alpha_mask.shape
    if use_cluster:
        xyz, cluster_labels = cluster_points(xyz, 3, 50)

        for label in np.unique(cluster_labels):
            alpha_mask = np.zeros_like(alpha_mask)
            cur_obj_xyz = xyz[cluster_labels == label]
            alpha_mask[cur_obj_xyz[:, 0], cur_obj_xyz[:, 1], cur_obj_xyz[:, 2]] = True

            vertices, triangles = mcubes.marching_cubes(alpha_mask, 0.5)
            vertices = vertices * voxel_size + xyz_min_fine
            mcubes.export_obj(vertices, triangles, f'scene_mesh_{label}.obj')
    else:
        vertices, triangles = mcubes.marching_cubes(alpha_mask, 0.5)
        vertices = vertices * voxel_size + xyz_min_fine
        mcubes.export_obj(vertices, triangles, path.replace('.npz', '.obj'))

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path')
    parser.add_argument('thres', type=float)
    parser.add_argument('--cam')
    parser.add_argument("--use_cluster", action='store_true')
    args = parser.parse_args()


    if os.path.isdir(args.path):
        for npz_path in tqdm(glob(os.path.join(args.path, "*/scene_mesh.npz"))):
            run(npz_path, args.use_cluster)
    else:
        run(args.path, args.use_cluster)

