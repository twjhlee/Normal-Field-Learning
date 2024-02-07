_base_ = '../../default.py'

basedir = './logs/nfl_example'

data = dict(
    datadir='data/nfl_example_dataset',
    dataset_type='inmc',
    white_bkgd=False,
    near_far=[0.1, 2], # - for real
    bg_color = [0.0, 0.0, 0.0],
    train_step=1,
    fixate_voxel=False,
    world_bbox = [-0.3, -0.3, -0.3, 0.3, 0.3, 0.3], # world bounding box, [x1, y1, z1, x2, y2, z2] - for dexnerf
    use_mask=False,
    use_kappa=False,
    use_bernoulli=False,
    mask_thresh=0.5,
    load2gpu_on_the_fly=False  
)

coarse_train = dict(
    N_rand=8192,
    weight_rgbper=0,
    weight_entropy_last=0.0,     # weight of background entropy loss
    weight_bg_density=10.0,
    N_iters=0,
    lrate_density=0.1,
    lrate_k0=0.1,
    weight_tv_density=1e-4,      # total variation loss for density
    weight_tv_k0=1e-7,           # total variation loss for normal color
    tv_before=5000,                  # count total variation before the given number of iterations
)

coarse_model_and_render = dict(
    loss_type='angular',
    num_voxels=50**3,
    num_voxels_base=50**3,
)

fine_train = dict(
    N_iters=4000,
    N_rand=32768,
    ray_sampler='random',         # ray sampling strategies
    weight_rgbper=0,
    weight_entropy_last=0.0,     # weight of background entropy loss
    weight_bg_density=0.01,
    lrate_density=0.1,
    lrate_k0=0.1,
    weight_tv_density=1e-5,      # total variation loss for density
    weight_tv_k0=1e-5,           # total variation loss for normal color
    tv_before=10000,                  # count total variation before the given number of iterations
    tv_dense_before=10000,            # count total variation densely before the given number of iterations
)

fine_model_and_render = dict(
    loss_type='angular',
    num_voxels=200**3,
    num_voxels_base=200**3,
    stepsize=0.4,                 # sampling stepsize in volume rendering
    rgbnet_dim=0,
)

