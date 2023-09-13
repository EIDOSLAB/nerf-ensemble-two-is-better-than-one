_base_ = 'default.py'

expname = 'dvgo_lego_275step'
basedir = '/scratch/nerf/NeRF-Ensemble/models'

data = dict(
    datadir='/scratch/nerf/dataset/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=True,
)
