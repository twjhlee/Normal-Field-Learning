import argparse
import numpy as np
import open3d as o3d
import cv2
from cluster_points import cluster_points
from scipy.spatial.transform import Rotation as R
from matplotlib import cm
import matplotlib as plt

#region geometry utils

ARROW_DIR = np.array([0, 0, 1]) # o3d arrow default direction. [http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html]
BG_DIR = np.array([0, 0, -1])

fix_pose_mat = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1],
])

def align_vectors(a, b):
    b = b / np.linalg.norm(b) # normalize a
    a = a / np.linalg.norm(a) # normalize b
    v = np.cross(a, b)
    # s = np.linalg.norm(v)
    c = np.dot(a, b)

    v1, v2, v3 = v
    h = 1 / (1 + c)

    Vmat = np.array([[0, -v3, v2],
                  [v3, 0, -v1],
                  [-v2, v1, 0]])

    R = np.eye(3, dtype=np.float64) + Vmat + (Vmat.dot(Vmat) * h)
    return R

def rgb2normal(rgb, fix_pose=False):
    '''
    convert rgb [0, 1] to normal [-1, 1]
    '''
    normals = rgb * 2 - 1
    normals /= np.linalg.norm(normals, axis=-1)[..., None]
    normals[np.isnan(normals)] = 1
    if fix_pose:
        normals = np.matmul(normals, fix_pose_mat)
    # normals[..., 1:] *= -1 # gnd2world
    return normals

def create_arrow(origin=np.array([0,0,0]),
                 direction=np.array([0,0,1]),
                 scale_factor=1.0,
                 color=None,
                 **kwargs):
    '''
    origin:     starting point of cylinder
    direction:  arrow direction
    color:      [DEPRECATED] rgb color
    '''
    arrow_vis_option = dict(
        cylinder_radius=0.4 * scale_factor,
        cone_radius=0.6 * scale_factor,
        cylinder_height=2.0 * scale_factor,
        cone_height=1.0 * scale_factor,
    )
    arrow_vis_option.update(kwargs)
    arrow = o3d.geometry.TriangleMesh.create_arrow(**arrow_vis_option)
    if color is None:
        arrow_vertices = np.asarray(arrow.vertices)
        arrow_colors = cm.jet(arrow_vertices[:, -1] / (alpha.shape[-1] - 25))[:, :-1]

        arrow.vertex_colors = o3d.utility.Vector3dVector(arrow_colors)
    else:
        arrow.paint_uniform_color(color)

    rot_mat = align_vectors(ARROW_DIR, direction)
    arrow.rotate(rot_mat, np.array([0,0,0]))
    arrow.translate(origin)

    return arrow

#endregion geometry utils

#region visualizer utils
is_black = False
def change_background_to_black(vis):
    global is_black
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1] if is_black else [0, 0, 0])
    is_black = ~is_black
    return False

GEOMETRY_IDX = 0
def show_next_geometry(vis):
    global GEOMETRY_IDX
    geometry = geometries[GEOMETRY_IDX]
    GEOMETRY_IDX = (GEOMETRY_IDX + 1) % len(geometries)
    vis.clear_geometries()
    vis.add_geometry(geometry)
    vis.update_renderer()
    return False

def show_all_geometry(vis):
    vis.clear_geometries()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.update_renderer()
    return False

def capture_depth(vis, path):
    depth = vis.capture_depth_float_buffer()
    plt.imshow(np.asarray(depth))
    plt.savefig(path, bbox_inches='tight', dpi=300)
    return False

def capture_image(vis, path):
    img = vis.capture_screen_float_buffer()
    plt.imshow(np.asarray(img))
    plt.savefig(path, bbox_inches='tight', dpi=300)
    return False

def load_render_option(vis):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters("capture_pose.json")
    ctr.convert_from_pinhole_camera_parameters(param)
    return False

def save_render_option(vis):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("capture_pose.json", param)
    return False


def run_visualizer(geometries):
    # create visualizer
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(left=0, top=0, height=1080, width=1920)
    vis.register_key_callback(ord('K'), change_background_to_black)
    vis.register_key_callback(ord('N'), show_next_geometry)
    vis.register_key_callback(ord('A'), show_all_geometry)
    vis.register_key_callback(ord('R'), load_render_option)
    vis.register_key_callback(ord('S'), save_render_option)
    for geom in geometries:
        vis.add_geometry(geom)
    # show_next_geometry(vis)
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=90)
    vis.run()

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('path')
    parser.add_argument('thres', type=float)
    parser.add_argument('--cam')
    parser.add_argument("--x_min", type=int, default=0)
    parser.add_argument("--x_max", type=int, default=500)
    parser.add_argument("--y_min", type=int, default=0)
    parser.add_argument("--y_max", type=int, default=500)
    parser.add_argument("--z_min", type=int, default=0)
    parser.add_argument("--z_max", type=int, default=500)
    parser.add_argument("--visualize_kappa", default=0)
    parser.add_argument("--remove_floater", action='store_true')
    parser.add_argument("--fix_pose", action='store_true')
    parser.add_argument("--extract_surface", action='store_true')
    parser.add_argument("--write_point_cloud", action='store_true')
    parser.add_argument("--cube", action='store_true')
    parser.add_argument("--run_vis", action='store_true')
    # parser.add_argument('--grasp_data', type=str, required=True)
    args = parser.parse_args()

    data = np.load(args.path)
    alpha = data['alpha']
    rgb = data['rgb']
    kappa = data['kappa']

    # Cut to desired parts
    x_max = min(rgb.shape[0], int(args.x_max))
    y_max = min(rgb.shape[1], int(args.y_max))
    z_max = min(rgb.shape[2], int(args.z_max))
    x_min, y_min, z_min = int(args.x_min), int(args.y_min), int(args.z_min)

    alpha = alpha[x_min: x_max, y_min: y_max, z_min: z_max]
    rgb = rgb[x_min: x_max, y_min: y_max, z_min: z_max]
    kappa = kappa[x_min: x_max, y_min: y_max, z_min: z_max]

    if rgb.shape[0] < rgb.shape[-1]:
        alpha = np.transpose(alpha, (1,2,0))
        rgb = np.transpose(rgb, (1,2,3,0))

    alpha_mask = alpha > args.thres

    print('Shape', alpha.shape, rgb.shape)
    print('Active rate', alpha_mask.mean())
    print('Active nums', alpha_mask.sum())
    xyz_min = np.array([0,0,0])
    xyz_max = np.array(alpha.shape)

    if args.cam:
        data = np.load(args.cam)
        xyz_min = data['xyz_min']
        xyz_max = data['xyz_max']
        cam_lst = data['cam_lst']
        cam_frustrm_lst = []
        for cam in cam_lst:
            cam_frustrm = o3d.geometry.LineSet()
            cam_frustrm.points = o3d.utility.Vector3dVector(cam)
            if len(cam) == 5:
                cam_frustrm.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for i in range(8)])
                cam_frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]])
            elif len(cam) == 8:
                cam_frustrm.colors = o3d.utility.Vector3dVector([[0.5,0.5,0.5] for i in range(12)])
                cam_frustrm.lines = o3d.utility.Vector2iVector([
                    [0,1],[1,3],[3,2],[2,0],
                    [4,5],[5,7],[7,6],[6,4],
                    [0,4],[1,5],[3,7],[2,6],
                ])
            cam_frustrm_lst.append(cam_frustrm)
    else:
        cam_frustrm_lst = []


    aabb_01 = np.array([[0, 0, 0],
                        [0, 0, 1],
                        [0, 1, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0]])
    out_bbox = o3d.geometry.LineSet()
    out_bbox.points = o3d.utility.Vector3dVector(xyz_min + aabb_01 * (xyz_max - xyz_min))
    out_bbox.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
    out_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

    xyz = np.stack(alpha_mask.nonzero(), -1)
    if args.remove_floater:
        xyz, cluster_labels = cluster_points(xyz, 3, 50)

    color = rgb[xyz[:,0], xyz[:,1], xyz[:,2]]
    kappa_ = kappa[xyz[:,0], xyz[:,1], xyz[:,2]]
    # normal = color * 2 - 1
    normal = rgb2normal(color, args.fix_pose)
    # normal[..., 2] *= -1

    # create geometry
    geometries = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz / alpha.shape * (xyz_max - xyz_min) + xyz_min)
    pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
    if args.write_point_cloud:
        pcd_name = args.path.replace("scene_mesh.npz", "pointcloud.ply")
        o3d.io.write_point_cloud(pcd_name, pcd)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=max((xyz_max - xyz_min) / alpha.shape) * 1.0)
    geometries.append(voxel_grid)
    # Cube surrounding workspace

    # Dexnerf scene 
    origin_point = [0, 15, 0]
    grid_x = 130
    grid_y = 130
    grid_z = 70

    # Blender scene
    origin_point = [30, 30, 20]
    grid_x = 120
    grid_y = 80
    grid_z = 50

    # Our scene
    # origin_point = [90, 85, 0]
    # grid_x = 100
    # grid_y = 105
    # grid_z = 65
 
    
    points = [
        [origin_point[0] , origin_point[1], origin_point[2]],
        [origin_point[0] + grid_x , origin_point[1], origin_point[2]],
        [origin_point[0] , origin_point[1] + grid_y, origin_point[2]],
        [origin_point[0] + grid_x , origin_point[1] + grid_y, origin_point[2]],
        [origin_point[0] , origin_point[1], origin_point[2] + grid_z],
        [origin_point[0] + grid_x , origin_point[1], origin_point[2] + grid_z],
        [origin_point[0] , origin_point[1] + grid_y, origin_point[2] + grid_z],
        [origin_point[0] + grid_x , origin_point[1] + grid_y, origin_point[2] + grid_z],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[0.2, 0.2, 0.2] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    if args.cube:
        geometries.append(line_set)

    if args.run_vis:
        run_visualizer(geometries)
