'''util functions for grasping'''
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from einops import rearrange

def _is_valid_grasp(x1:np.ndarray,
                   x2:np.ndarray,
                   n1:np.ndarray,
                   n2:np.ndarray,
                   minimum_distance:float=0.0,
                   maximum_distance:float=5.0,
                   normal_threshold:float=0.9,
                   distortion_threshold:float=0.9
                   )->np.ndarray:
    '''
    x1, x2, n1, n2: [N, 3]
    return:         [N, 1] boolean array
    '''
    assert x1.shape == x2.shape == n1.shape == n2.shape
    distance = np.linalg.norm(x1 - x2, axis=1)
    is_valid_distance = (minimum_distance <= distance) & (distance <= maximum_distance)

    # sift out distant samples
    x1, x2, n1, n2, distance = x1[is_valid_distance], x2[is_valid_distance], n1[is_valid_distance], n2[is_valid_distance], distance[is_valid_distance]
    # is_valid_normal = np.zeros_like(is_valid_distance, dtype=bool)
    # is_valid_normal[is_valid_distance] = ((n1 * n2).sum(axis=-1) <= -np.abs(normal_threshold))

    # outward direction
    is_affordable_distortion = np.zeros_like(is_valid_distance, dtype=bool)
    position_vec = (x2 - x1) / distance[:, None]
    is_affordable_distortion[is_valid_distance] = ((position_vec * n1).sum(axis=-1) <= -distortion_threshold) & \
                               ((position_vec * n2).sum(axis=-1) >= distortion_threshold) 

    # return is_valid_distance & is_valid_normal & is_affordable_distortion
    return is_valid_distance & is_affordable_distortion

def find_grasping_candidate(points:np.ndarray,
                            normals:np.ndarray, 
                            cluster_labels:np.ndarray,
                            priorities:np.ndarray,
                            min_gripper_width:float=5.0,
                            max_gripper_width:float=20.0,
                            normal_threshold:float=0.9,
                            distortion_threshold:float=0.9,
                            num_candidates:int=10
                            )->np.ndarray:
    candidates = []

    # sort by priorities, in particular it can be a sigma
    # sort_idx = np.argsort(priorities)
    sort_idx = np.argsort(points[:, 2])[::-1]
    points, normals, priorities, cluster_labels = points[sort_idx], normals[sort_idx], priorities[sort_idx], cluster_labels[sort_idx]

    for i, (point, normal) in tqdm(enumerate(zip(points, normals))):
        if i >= len(points) or len(candidates) >= num_candidates:
            break
       
        target_cluster = cluster_labels[i]
        cluster_flags = cluster_labels[i+1:] == target_cluster
        x2 = points[i+1:][cluster_flags]
        n2 = normals[i+1:][cluster_flags]
        
        num_remain_points = cluster_flags.sum()
        x1 = np.repeat(point[None, :], num_remain_points, axis=0) # [N-i-1, 3]
        n1 = np.repeat(normal[None, :], num_remain_points, axis=0) # [N-i-1, 3]

        grasp_flags = _is_valid_grasp(x1, x2, n1, n2, minimum_distance=min_gripper_width, maximum_distance=max_gripper_width,
                                      normal_threshold=normal_threshold, distortion_threshold=distortion_threshold)
        
        p1 = np.repeat(priorities[i], num_remain_points, axis=0) # [N-i-1]
        p2 = priorities[i+1:][cluster_flags]

        if grasp_flags.sum() > 0:
            valid_idxs = np.nonzero(grasp_flags)[0]
            priority = p1[valid_idxs] + p2[valid_idxs]
            target_cluster_array = np.repeat(target_cluster, len(valid_idxs))
            candidates.append(np.concatenate([x1[valid_idxs], x2[valid_idxs], n1[valid_idxs], n2[valid_idxs], priority[:, None], target_cluster_array[:, None]], axis=1))
    
    return np.concatenate(candidates)


@torch.no_grad()
def sample_grasp(normal_grid:torch.Tensor,
                 alpha_grid:torch.Tensor,
                 alpha_thres=0.7,
                 num_candidates=1024,
                 normal_thres=0.1,
                 min_grasp_dist=0.05,
                 max_grasp_dist=0.2,
                 num_samples=10,            
                 voxel_scale=1.,
                 ):
    '''Sample grasping candidates from normal grid
    normal_grid:    3D normal grid, [H, W, D, 3]
    occupancy_grid: 3D occupancy grid for skipping samples. [H, W, D]
    num_candidates: random sample candidates
    normal_thres:   gaurantee two normals are opposite directions
    min_grasp_dist: minimum distance for grasping
    max_grasp_dist: maximum distance for grasping
    num_samples:    grasping sample count between [min_grasp_dist, max_grasp_dist]
    voxel_scale:    real-world scale of voxel[m]
    '''
    assert normal_grid.shape[:3] == alpha_grid.shape[:3]
    H, W, D = normal_grid.shape[:3]
    
    # sample points from occupied voxels
    occupancy_grid = alpha_grid > alpha_thres

    normal_grid = normal_grid * occupancy_grid[..., None]
    
    occupied_index = occupancy_grid.nonzero()
    num_candidates = min(num_candidates, len(occupied_index))

    # occupied_index = occupied_index[alpha_grid[occupied_index[:, 0], occupied_index[:, 1], occupied_index[:, 2]].argsort(descending=True)[:num_candidates]]

    occupied_index = occupied_index[torch.randperm(len(occupied_index))[:num_candidates]]
    normal_dirs = normal_grid[occupied_index[:, 0], occupied_index[:, 1], occupied_index[:, 2]]
    alphas = alpha_grid[occupied_index[:, 0], occupied_index[:, 1], occupied_index[:, 2]]

    # sample points along the opposite normal directions
    min_dist, max_dist = min_grasp_dist / voxel_scale, max_grasp_dist / voxel_scale
    sample_dist = torch.linspace(min_dist, max_dist, num_samples).to(normal_dirs)
    sample_dist = torch.einsum('ij,k->ikj', normal_dirs, sample_dist)

    sample_index = occupied_index.unsqueeze(dim=1) - sample_dist # minus sign stands for opposite direction
    sample_grid = rearrange(sample_index, 'n s c -> 1 1 1 (n s) c') # [1, 1, 1, N*S, 3]
    sample_grid = (sample_grid / torch.Tensor([H, W, D]).to(normal_dirs)) * 2 - 1 # [-1, 1]

    normal_grid = rearrange(normal_grid, 'h w d c -> 1 c h w d') # [1, C, H, W, D]
    sample_normals = F.grid_sample(normal_grid, sample_grid, align_corners=True, padding_mode='border')
    sample_normals = rearrange(sample_normals, '1 c 1 1 (n s) -> n s c', n=num_candidates)

    alpha_grid = rearrange(alpha_grid, 'h w d -> 1 1 h w d') # [1, 1, H, W, D]
    sample_alphas = F.grid_sample(alpha_grid, sample_grid, align_corners=True, padding_mode='border')
    sample_alphas = rearrange(sample_alphas, '1 1 1 1 (n s) -> n s 1', n=num_candidates)

    # normalize sampled normals
    sample_normals = sample_normals / sample_normals.norm(dim=2)[..., None] # [N, S, C]
    sample_normals[sample_normals.isnan()] = 0

    # check the normals are in the opposite direction
    # align_sample_index = ((normal_dirs.unsqueeze(1) + sample_normals).norm(dim=2) < normal_thres).nonzero()
    align_sample_index = ((normal_dirs.unsqueeze(1) * sample_normals).sum(dim=2) < (-1 + normal_thres)).nonzero()
    unique_index = align_sample_index[:, 0].unique()   

    # get points and normals for each grasping points
    x1 = occupied_index[unique_index]
    n1 = normal_dirs[unique_index]
    alpha1 = alphas[unique_index]

    _x2_index = align_sample_index[torch.stack([(align_sample_index[:, 0] == i).nonzero()[-1] for i in unique_index]).squeeze()]
    x2 = sample_index[_x2_index[:, 0], _x2_index[:, 1]]
    n2 = sample_normals[_x2_index[:, 0], _x2_index[:, 1]]
    alpha2 = sample_alphas[_x2_index[:, 0], _x2_index[:, 1]].squeeze()
    
    # sort_index = (alpha1 + alpha2).argsort(descending=True)
    sort_index = alpha2.argsort(descending=True)

    x1, n1, x2, n2 = x1[sort_index], n1[sort_index], x2[sort_index], n2[sort_index]
    
    return x1, n1, x2, n2