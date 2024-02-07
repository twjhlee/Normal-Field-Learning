_base_ = './default_nfl_example.py'
expname = 'norm_kappa_fixate'

data = dict(
    datatype = 'pred_normal_pred_masked',             # rgb | normal | pred_mask
    fixate_voxel=True,
    use_mask=False,
    use_kappa=True,
    use_bernoulli=False
)