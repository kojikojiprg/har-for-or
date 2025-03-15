# HAR for Operating Room Surveillance Video
A project designed to apply activity recognition models to operating room surveillance footage, aiming to enhance situational awareness and detect critical events during surgeries.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# Overview
This project focuses on utilizing human activity recognition (HAR) models to analyze operating room (OR) surveillance videos. By processing video data, the system aims to recognize activities, enhance operational safety, and identify crucial events in real-time. The project explores the intersection of computer vision, machine learning, and healthcare improvement.

# Environments
- Ubuntu: 20.04
- Python: 3.10.14
- CUDA: 12.1

# Setup
```
pip install -U pip
pip install -r requirements/requirements-run.txt
pip install -r requirements/requirements-model.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/requirements-mmcv.txt --no-cache-dir
pip install -r requirements/requirements-develop.txt
```

# Preparing Dataset
Extract individual keypoints from operating room surveillance videos using object detection (YOLOv8), pose estimation (ViTPose), and tracking (BoostTrack+).
Due to apply our models for long time length videos, it writes shards by splitting dataset for Itterable Dataset of Pytorch.

```
python write_shards.py [-c CONFIG_DIR] [-cht CONFIG_HUMAN_TRACKING_PATH] [-np N_PROCESSES] [-g GPU_ID] data_root dataset_type
```

### Positional Arguments:
  - data_root:            The root directory of the dataset
  - dataset_type:         'individual' or 'group'
    - 'individual': Write the time-series bounding boxes and keypoints of each individual in into shards.
    - 'group': Write the time-series bounding boxes and keypoints of all captured individuals in each frame into shards (Not Implemented Models).

### Options:
  - -c, --config_dir:     The directory of configs.
  - -cht, --config_human_tracking_path CONFIG_HUMAN_TRACKING_PATH:  
    The path of human tracking config path.
  - -np, --n_processes N_PROCESSES:  
    The number of processes for multiprocessing.
  - -g, --gpu_id GPU_ID:  
    The id of gpu which you want to use.




# Training Models
## Training Categorical SQ-VAE
```
python train_individual.py [-ckpt CHECKPOINT] [-g [GPU_IDS ...]] [data_root_lst ...]
```

### Positional Arguments:
  - data_root_lst:  
    The list of the datasets which you want to use training.

### Options:
  - -ckpt, --checkpoint CHECKPOINT:  
    The pretrained checkpoint path (*.ckpt) to continue training.
  - -g, --gpu_ids [GPU_IDS ...]:  
    The ids of gpu which you want to use.

## Training Diffusion Model
```
python train_individual_diffusion.py -ckpt CSQVAE_CHECKPOINT [-g [GPU_IDS ...]] [data_root_lst ...]
```

### Positional Arguments:
  - data_root_lst:  
    The list of the datasets which you want to use training.

### Options:
  - -ckpt, --csqvae_checkpoint CSQVAE_CHECKPOINT:  
    The pretrained checkpoint path of CSQ-VAE model (*.ckpt) that yoou want to generate human activities.
  - -g, --gpu_ids [GPU_IDS ...]:  
    The ids of gpu which you want to use.

# Prediction
```
pred_individual.py [-v VERSION] [-g GPU_ID] data_root
```

positional arguments:
  - data_root: The root directory of the dataset for predict role classes of individiauls.

options:
  - -v, --version VERSION: The model version.
  - -g, --gpu_id GPU_ID:  
    The id of gpu which you want to use.

# Reference
The paper of Categorical SQ-VAE for individual activity recognition was accepted by International Journal of Activity and Behavior Computing.
Coming soon!
```

```
