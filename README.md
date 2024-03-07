# group_activity

# Environments
- Ubuntu: 20.04
- Python: 3.9.15
- CUDA: 12.0

# Setup
```
pip install -U pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

# Prepairing Bounding Boxes and Keypoints
We extract individual bounding boxs and keypoynts using [this repository](https://github.com/kojikojiprg/pose_estimation), which is used for our research projects.

# Optical Flow
Calcurate the oprtical flow from .mp4 videos.
The output file 'flow.npy' will be stored into the dataset directory.

```
python tools/optical_flow.py [-dt DATASET_TYPE] [--comp] dataset_dir
```

positional arguments:
  dataset_dir           path of input dataset directory

optional arguments:
  - --comp                compress output from float32 to float16.

# Training Models
# Prediction
# Reference
