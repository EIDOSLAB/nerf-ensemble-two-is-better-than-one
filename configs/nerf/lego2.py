_base_ = 'default2.py'

expname = 'dvgo_lego_80_black'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=False,
)

