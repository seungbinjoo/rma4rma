Repo for Rapid Motor Adaptation for Robotic Manipulator Arms ([link]([www.google.com](https://arxiv.org/abs/2312.04670))). It's still under construction.

# Installation
1. Clone the repo:
```
git clone --recurse-submodules https://github.com/yichao-liang/rma2
```

2. Create a conda environment:
```
conda env create -f environment_copy.yml
```

# Example code
1. Base policy training:
```
python main.py -n 50 -bs 5000 -rs 2000 \
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 
```

2. Adaptation training:
```
python train_sb.py -n 50 -bs 5000 -rs 2000 \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --adaptation_training \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1 \
    --ckpt_name best_model.zip
```

3. Evaluate:
```
python train_sb.py --eval \
    --randomized_training --ext_disturbance --obs_noise --use_depth_adaptation\
    -e PickSingleYCB-v1 --ckpt_name best_model.zip \
    --log_name PPO-ps-bs5000-rs2000-kl0.05-neps10-cr0.2-lr_scdl0-cr_scdl1-ms50-incObsN0_1_stage2_dep_1
```