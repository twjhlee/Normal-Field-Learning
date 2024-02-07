_base_ = './default_nfl_example.py'
expname = 'DVGO'

data = dict(
    # datatype = 'rgb_pred_masked',             # rgb | normal | pred_mask
    datatype = 'rgb',             # rgb | normal | pred_mask
    near_far=[0.2, 3],
    fixate_voxel=False,
    use_mask=False,
    use_kappa=False,
    use_bernoulli=False,
    render_depth_sigma_thres=0,
)

coarse_model_and_render = dict(
    loss_type='mse'
)

coarse_train = dict(
    weight_rgbper=0,
    N_rand=8192,
    weight_entropy_last=0.0,     # weight of background entropy loss
    weight_bg_density=0.0,
    N_iters=5000,
    lrate_density=0.1,
    lrate_k0=0.1,
    weight_tv_density=1e-7,      # total variation loss for density
    weight_tv_k0=1e-7,           # total variation loss for normal color
    tv_before=5000,                  # count total variation before the given number of iterations
)

fine_train = dict(
    N_iters=20000,
    N_rand=8192,
    weight_bg_density=0.001,
    weight_tv_density=1e-7,
    weight_tv_k0=1e-7
)
fine_model_and_render = dict(
    loss_type='mse',
    rgbnet_dim=12,
)