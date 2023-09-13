_base_ = 'default.py'

expname = 'dvgo_drums_300_white'
basedir = '/scratch/nerf/Multi-NeRF/models'

data = dict(
    datadir='/scratch/nerf/dataset/nerf_synthetic/drums',
    dataset_type='blender',
    white_bkgd=True,
)
