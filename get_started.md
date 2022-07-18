# Get Started
## Prerequisite

1. mmcv==1.4.0 (https://github.com/open-mmlab/mmcv)
2. mmdet (https://github.com/open-mmlab/mmdetection)
3. mmseg (https://github.com/open-mmlab/mmsegmentation)
4. mmdet3d==1.17.1 (https://github.com/open-mmlab/mmdetection3d)

## Dataset
1. Following [mmdet3d](https://github.com/open-mmlab/mmdetection3d) to process the data.

## Train
1. Following [DETR3D](https://github.com/WangYueFt/detr3d) to get FCOS pretrain weight.
2. To train our major model, please use
```
./tools/dist_train.sh projects/configs/ego3rt/ego3rt_polar80x256_cart160x160.py 8
```