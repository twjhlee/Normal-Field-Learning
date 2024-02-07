import argparse
import numpy as np
from tqdm import tqdm
import mcubes
from cluster_points import cluster_points
import os
from grasp_util import find_grasping_candidate

fix_pose_mat = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
])

def rgb2normal(rgb):
    '''
    convert rgb [0, 1] to normal [-1, 1]
    '''
    normals = rgb * 2 - 1
    normals /= np.linalg.norm(normals, axis=-1)[..., None]
    normals = np.matmul(normals, fix_pose_mat)
    normals[np.isnan(normals)] = 1
    # normals[..., 1:] *= -1 # gnd2world
    return normals

def write_urdf(save_urdf_path, template_path):
    old = 'scene_mesh.obj'
    new = os.path.basename(save_urdf_path.replace('.urdf', '.obj'))
    with open(save_urdf_path, 'w') as save_file:
        with open(template_path, 'r') as template_file:
            for line in template_file:
                save_file.write(line.replace(old, new))
                
def main(args):
    URDF_TEMPLATE_PATH = './scene_mesh.urdf'

    data = np.load(args.path)
    alpha = data['alpha']
    rgb = data['rgb']

    xyz_min_fine = np.array([0.2, -0.3, 0.02])
    xyz_max_fine = np.array([0.8, 0.3, 0.3])

    # Cut to desired parts
    if rgb.shape[0] < rgb.shape[-1]:
        alpha = np.transpose(alpha, (1,2,0))
        rgb = np.transpose(rgb, (1,2,3,0))

    alpha_mask = alpha > args.thres

    print('Shape', alpha.shape, rgb.shape)
    print('Active rate', alpha_mask.mean())
    print('Active nums', alpha_mask.sum())

    xyz = np.stack(alpha_mask.nonzero(), -1)
    z_thres = 0.02 / (xyz_max_fine[2] - xyz_min_fine[2]) * alpha.shape[2]
    xyz = xyz[xyz[..., 2] > z_thres]
    xyz, cluster_labels = cluster_points(xyz, 4, 50)

    color = rgb[xyz[:,0], xyz[:,1], xyz[:,2]]

    # grasping candidate
    priorities = alpha[xyz[:, 0], xyz[:, 1], xyz[:, 2]]
    normals = rgb2normal(color)

    world_to_vox = alpha.shape[0] / (xyz_max_fine[0] - xyz_min_fine[0])

    min_gripper_width, max_gripper_width = 0.03, 0.075 # world scale [m]
    min_gripper_width, max_gripper_width = int(min_gripper_width * world_to_vox), int(max_gripper_width * world_to_vox)

    print('min_width: {} , max_width: {}'.format(min_gripper_width, max_gripper_width))

    candidates = find_grasping_candidate(xyz, normals, cluster_labels, priorities, 
                                        min_gripper_width=min_gripper_width,
                                        max_gripper_width=max_gripper_width,
                                        num_candidates=1000,
                                        normal_threshold=0.95,
                                        distortion_threshold=0.99)

    candidates = candidates[candidates[:, 12].argsort()[::-1]]

    # top-K selection
    target = candidates[:100]

    x1, x2, n1, n2, target_cluster = target[:, 0:3], target[:, 3:6], target[:, 6:9], target[:, 9:12], target[:, 13]

    world_x1 = xyz_min_fine + (xyz_max_fine - xyz_min_fine) * x1 / alpha.shape
    world_x2 = xyz_min_fine + (xyz_max_fine - xyz_min_fine) * x2 / alpha.shape

    grasp_data_fname = os.path.join(os.path.dirname(args.path), 'grasp_data')
    np.savez_compressed(grasp_data_fname,
                        x1=world_x1,
                        x2=world_x2,
                        n1=n1,
                        n2=n2,
                        target_cluster=target_cluster,
                        total_cluster_cnt=len(np.unique(cluster_labels)))

    scene_size = xyz_max_fine - xyz_min_fine
    voxel_size = scene_size / alpha_mask.shape

    for label in np.unique(cluster_labels):
        alpha_mask = np.zeros_like(alpha_mask)
        cur_obj_xyz = xyz[cluster_labels == label]
        alpha_mask[cur_obj_xyz[:, 0], cur_obj_xyz[:, 1], cur_obj_xyz[:, 2]] = True

        vertices, triangles = mcubes.marching_cubes(alpha_mask, 0.5)
        vertices = vertices * voxel_size + xyz_min_fine
        
        mcubes.export_obj(vertices, triangles, os.path.join(os.path.dirname(args.path), f'scene_mesh_{label}.obj'))
        write_urdf(os.path.join(os.path.dirname(args.path), f'scene_mesh_{label}.urdf'), URDF_TEMPLATE_PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path')
    parser.add_argument('thres', type=float)

    args = parser.parse_args()

    main(args)