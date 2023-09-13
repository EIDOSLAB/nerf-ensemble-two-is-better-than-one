# nerf-ensemble-two-is-better-than-one

This repo contains the implementation for the paper "Two is better than one: achieve high quality reconstruction with a NeRF ensemble".This work has been accepted to the International Conference on Image Analysis and Processing (ICIAP) 2023. 

### How to Run
The project consists of three main files:
1. *run_dvgo.py*  -> original DVGO implementation
2. *train_single_model.py* -> just a wrapper for easily running dvgo experiments;
3.  *main.py* -> the main program of our project

In order to make all work, you have to train *N* models *independently* by executing *train_single_model.py*. Inside the code, you can specify the dataset and the scene you want to train the models with (be sure to download the datasets and check their paths). 
Then you can simply execute:
    main.py --config configs/nerf/lego.py --ckpts *path_to_model_1*, *path_to_model_2*
if you want to also compress the models during training (Conjoint Pruning of the Ensemble, CPE), you can execute:
    main.py --config configs/nerf/lego.py --ckpts *path_to_model_1*, *path_to_model_2* --renerf
