_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters=True
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

model = dict(
    type='Ego3RT',
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        strides=(1, 2, 2, 2),
        dilations=(1, 1, 1, 1),
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=norm_cfg,
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    img_neck=dict(
        type='Ego3rtTracing',
        d_model=256,
        nhead=8,
        bev_shape=128, 
        polar_size=(72, 192),
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=3,
        dec_n_points=3,
        enc_n_points=3,
        topdown_layers=8,
        norm_cfg=norm_cfg,
        pc_range=point_cloud_range,
        topdown_cfg=dict(
            type='Bottleneck',
            block_cfg=dict(
                inplanes=256,
                planes=256,
                norm_cfg=norm_cfg,
                dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False))
            ),
        ),
    pts_bbox_head=dict(
        type='Ego3RTCenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['pedestrian']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['barrier', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=4,
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            code_size=9, 
            ),
        separate_head=dict(
            type='SeparateHead', init_bias=-8, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[[2], [2.5, 5.5], [1.5, 5], [0.7], [1.5, 0.8], [1.2, 0.8]],
            score_threshold=0.1,
            pc_range=[-51.2, -51.2],
            out_size_factor=4,
            voxel_size=voxel_size[:2],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)))


class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'pedestrian',
    'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ScalePadMultiViewImage', virtual_img_size=(300, 200),size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                        'depth2img', 'cam2img', 'pad_shape',
                        'scale_factor', 'flip', 'pcd_horizontal_flip',
                        'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                        'img_norm_cfg', 'pcd_trans', 'sample_idx',
                        'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                        'transformation_3d_flow', 'cam_intrinsic', 'lidar2cam', 'virtual_cam_intrinsic'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ScalePadMultiViewImage', virtual_img_size=(300, 200),size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                        'depth2img', 'cam2img', 'pad_shape',
                        'scale_factor', 'flip', 'pcd_horizontal_flip',
                        'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                        'img_norm_cfg', 'pcd_trans', 'sample_idx',
                        'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                        'transformation_3d_flow', 'cam_intrinsic', 'lidar2cam', 'virtual_cam_intrinsic'))
        ])
]

# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='ScalePadMultiViewImage', virtual_img_size=(300, 200),size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                        'depth2img', 'cam2img', 'pad_shape',
                        'scale_factor', 'flip', 'pcd_horizontal_flip',
                        'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                        'img_norm_cfg', 'pcd_trans', 'sample_idx',
                        'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                        'transformation_3d_flow', 'cam_intrinsic', 'lidar2cam', 'virtual_cam_intrinsic'))
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, classes=class_names, modality=input_modality))

optimizer = dict(
    type='AdamW', 
    lr=2.5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr=4e-5)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='pretrained/fcos3d_r101.pth'
