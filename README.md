# One Class One Click (OCOC): Quasi Scene-level Weakly Supervised Point Cloud Semantic Segmentation with Active Learning
## Introduction
Official code for the paper One Class One Click: Quasi Scene-level Weakly Supervised Point Cloud Semantic Segmentation with Active Learning.
OCOC is a active weakly supervised learning point cloud semantic segmentation method ([paper link](https://www.sciencedirect.com/science/article/pii/S0924271623002344)). 

![workflow](https://github.com/PuzoW/One-Class-One-Click/blob/main/doc/workflow.jpg)

![weak supervision](https://github.com/PuzoW/One-Class-One-Click/blob/main/doc/wsl.jpg)

![active learning](https://github.com/PuzoW/One-Class-One-Click/blob/main/doc/active_learning.jpg)

## Dependency

1. Create conda environment
```
conda create -n ococ python=3.8
conda activate ococ
```

2. Install the package using 
```
pip install -r requirement.txt
```

3. Please follow the <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KPConv</a> to compile the C++ extension modules in `cpp_wrappers`


## Train & test

### Train
```
python train_H3D.py
```
### Test
For validation set
```
python test_model -s "H3D/log_name" -m val
```
For test set
```
python test_model -s "H3D/log_name" -m test
```
### Evaluation
```
python cal_metrics.py
```



If you find our work useful in your 
research, please consider citing:

```
@article{WANG202389,
title = {One Class One Click: Quasi scene-level weakly supervised point cloud semantic segmentation with active learning},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {204},
pages = {89-104},
year = {2023},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2023.09.002},
url = {https://www.sciencedirect.com/science/article/pii/S0924271623002344},
author = {Puzuo Wang and Wei Yao and Jie Shao},
}
```
## Acknowledgment

Our code uses the <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KPConv</a> as the backbone network.

## License
Our code is released under MIT License (see LICENSE file for details).
