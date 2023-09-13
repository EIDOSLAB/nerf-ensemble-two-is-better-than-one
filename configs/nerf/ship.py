_base_ = 'default.py'

expname = 'dvgo_ship_300_white'
basedir = '/scratch/nerf/Multi-NeRF/models'

data = dict(
    datadir='/scratch/nerf/dataset/nerf_synthetic/ship',
    dataset_type='blender',
    white_bkgd=True,
)
