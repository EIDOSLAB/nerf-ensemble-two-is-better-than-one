_base_ = 'default.py'

expname = 'dvgo_hotdog_300_white'
basedir = '/scratch/nerf/Multi-NeRF/models'

data = dict(
    datadir='/scratch/nerf/dataset/nerf_synthetic/hotdog',
    dataset_type='blender',
    white_bkgd=True,
)
