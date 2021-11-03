dataset_type = 'UMA3DDataset'
data_root = './data/uma3d/data/'
class_names = ('small_buoy', 'tall_buoy', 'dock')
train_pipeline = [
    dict(
        type='LoadPointsFromPCD',
        coord_type='DEPTH',
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=False,
        with_seg_3d=False),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.2, 0.2],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('small_buoy', 'tall_buoy', 'dock')),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromPCD',
        coord_type='DEPTH',
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='IndoorPointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('small_buoy', 'tall_buoy', 'dock'),
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin'),
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='UMA3DDataset',
            data_root='./data/uma3d/data/',
            ann_file='./data/uma3d/data/uma3d_infos_train.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromPCD',
                    coord_type='DEPTH',
                    load_dim=3,
                    use_dim=[0, 1, 2]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_mask_3d=False,
                    with_seg_3d=False),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.2, 0.2],
                    scale_ratio_range=[0.95, 1.05],
                    translation_std=[0, 0, 0]),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=('small_buoy', 'tall_buoy', 'dock')),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            filter_empty_gt=False,
            classes=('small_buoy', 'tall_buoy', 'dock'),
            box_type_3d='DEPTH')),
    val=dict(
        type='UMA3DDataset',
        data_root='./data/uma3d/data/',
        ann_file='./data/uma3d/data/uma3d_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromPCD',
                coord_type='DEPTH',
                load_dim=3,
                use_dim=[0, 1, 2]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='IndoorPointSample', num_points=20000),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=('small_buoy', 'tall_buoy', 'dock'),
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=('small_buoy', 'tall_buoy', 'dock'),
        test_mode=True,
        box_type_3d='DEPTH'),
    test=dict(
        type='UMA3DDataset',
        data_root='./data/uma3d/data/',
        ann_file='./data/uma3d/data/uma3d_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromPCD',
                coord_type='DEPTH',
                load_dim=3,
                use_dim=[0, 1, 2]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='IndoorPointSample', num_points=20000),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=('small_buoy', 'tall_buoy', 'dock'),
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=('small_buoy', 'tall_buoy', 'dock'),
        test_mode=True,
        box_type_3d='DEPTH'))
evaluation = dict(pipeline=[
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                     'window', 'bookshelf', 'picture', 'counter', 'desk',
                     'curtain', 'refrigerator', 'showercurtrain', 'toilet',
                     'sink', 'bathtub', 'garbagebin'),
        with_label=False),
    dict(type='Collect3D', keys=['points'])
])
model = dict(
    type='VoteNet',
    backbone=dict(
        type='PointNet2SASSG',
        in_channels=3,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        type='VoteHead',
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss',
            reduction='sum',
            loss_weight=3.3333333333333335),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        num_classes=3,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=3,
            num_dir_bins=1,
            with_rot=True,
            mean_sizes=[[1.44913911, 1.44210823, 1.51068091],
                        [1.2720509, 1.29382019, 1.45258459],
                        [2.45731725, 2.80586288, 2.02049193]])),
    train_cfg=dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote'),
    test_cfg=dict(
        sample_mod='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True))
lr = 0.008
optimizer = dict(type='AdamW', lr=0.008, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[24, 32])
runner = dict(type='EpochBasedRunner', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=30,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/votenet_uma-3d-3class'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 1)
