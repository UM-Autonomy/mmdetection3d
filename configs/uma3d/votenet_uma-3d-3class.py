_base_ = [
    '../_base_/datasets/scannet-3d-18class.py',
    '../_base_/models/votenet.py',
    '../_base_/schedules/schedule_uma.py',
    '../_base_/default_runtime.py'
]
# TODO: do we need to change the schedule in the future?

dataset_type = 'UMA3DDataset'
data_root = './data/uma3d/data/'
class_names = ('small_buoy', 'tall_buoy', 'dock')

model = dict(
    backbone=dict(
        in_channels=3),  # exclude 4th dim from votenet paper
    bbox_head=dict(
        num_classes=3,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=3,
            num_dir_bins=1,
            with_rot=True,
            # mean_sizes from data/uma3d/get_mean_class_sizes.py
            mean_sizes=[[1.44913911, 1.44210823, 1.51068091],
                        [1.2720509,  1.29382019, 1.45258459],
                        [2.45731725, 2.80586288, 2.02049193]])))

train_pipeline = [
    dict(
        type='LoadPointsFromPCD',
        coord_type='DEPTH',  # this means xyz data, i think?
        load_dim=3,  # we have 3 dimensions: (x, y, z)
        use_dim=[0, 1, 2]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,  # we have 3D bounding boxes
        with_label_3d=True,  # we have ground truth labels
        with_mask_3d=False,  # no masks in our data
        with_seg_3d=False),  # no segmentation in our data
    # TODO: add additional preprocessing steps if desired
   dict(
       type='GlobalRotScaleTrans',
       rot_range=[-0.2, 0.2],
       scale_ratio_range=[0.95, 1.05],
       translation_std=[0, 0, 0]
   ),
#    # Do not use jitter! decreases performance
#    dict(
#       type='RandomJitterPoints',
#       jitter_std=[0.01, 0.01, 0.01],
#        clip_range=[-0.05, 0.05]
#    ),
   dict(
       type='RandomFlip3D',
       flip_ratio_bev_vertical=0.3

   ),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

# TODO: mostly just copied this from sunrgbd-3d-10class.py,
#   need to look into what each step means
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
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5,
            ),
            dict(type='IndoorPointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# TODO: specify 'eval_pipeline' attribute

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'uma3d_infos_train.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=False,
            box_type_3d='DEPTH')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'uma3d_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='DEPTH'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'uma3d_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='DEPTH'))

# TODO: specify 'evaluation' attribute

# yapf:disable
log_config = dict(interval=30)
# yapf:enable