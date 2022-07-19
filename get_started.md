# Get Started
## Our setting
1. `Pytorch=1.8.1`
2. `CUDA=11.2`
3. NVIDIA RTX A6000 48G
## Prerequisite
1. mmcv==1.4.0 (https://github.com/open-mmlab/mmcv)
2. mmdet (https://github.com/open-mmlab/mmdetection)
3. mmseg (https://github.com/open-mmlab/mmsegmentation)
4. mmdet3d==1.17.1 (https://github.com/open-mmlab/mmdetection3d)
5. Link `mmdetection3d` to `Ego3RT/mmdetection3d`.

## Dataset & pretrain
1. Following [mmdet3d](https://github.com/open-mmlab/mmdetection3d) to process the data.
2. Link `mmdetection3d/data/` to `Ego3RT/data/`
3. Following [DETR3D](https://github.com/WangYueFt/detr3d) to get FCOS pretrain weight.
4. Your working directory should be
```
Ego3RT
    |- build
    |- data
        |- nuscenes
    |- mmdetection3d
    |- pretrained
        |- fcos3d.pth
    |- projects
    |- tools
    |- setup.py
```

## Install
```
cd Ego3RT
python setup.py develop
```
## Train
1. To train our major model, please use
```
./tools/dist_train.sh projects/configs/ego3rt/ego3rt_polar80x256_cart160x160.py 8
```