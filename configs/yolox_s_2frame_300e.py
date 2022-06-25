optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,
    num_last_epochs=15,
    min_lr_ratio=0.05)
runner = dict(type='EpochBasedRunner', max_epochs=301)
checkpoint_config = dict(interval=10)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', num_last_epochs=15, interval=10, priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=None,
        momentum=0.0001,
        priority=49)
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = 'work_dirs/yolox_s_8x8_300e_coco/best_bbox_mAP_epoch_287.pth'
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=64)
img_scale = (640, 640)
# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'
model = dict(
    type='YOLOX2Frame',
    input_size=(640, 640),
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = 'data/Data/csv/coco/'
dataset_type = 'CocoDualSmokeDataset'
train_pipeline = [
    dict(type='MosaicTwoFrames', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomAffine4TwoFrames',
        scaling_ratio_range=(0.1, 2),
        border=(-320, -320)),
    dict(
        type='MixUp4TwoFrames',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug4TwoFrames'),
    dict(type='RandomFlip4TwoFrame', flip_ratio=0.5),
    dict(type='Resize4TwoFrame', img_scale=(640, 640), keep_ratio=True),
    dict(
        type='Pad4TwoFrames',
        pad_to_square=True,
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle4TwoFrame'),
    dict(
        type='Collect4TwoFrame',
        keys=['img', 'img_2', 'img_2_gt', 'gt_bboxes', 'gt_labels'])
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDualSmokeDataset',
        ann_file='data/Data/csv/coco/annotations/instances_train2017.json',
        img_prefix='data/Data/csv/coco/images/train2017/',
        pipeline=[
            dict(type='LoadTwoImagesFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='MosaicTwoFrames', img_scale=(640, 640), pad_val=114.0),
        dict(
            type='RandomAffine4TwoFrames',
            scaling_ratio_range=(0.1, 2),
            border=(-320, -320)),
        dict(
            type='MixUp4TwoFrames',
            img_scale=(640, 640),
            ratio_range=(0.8, 1.6),
            pad_val=114.0),
        dict(type='YOLOXHSVRandomAug4TwoFrames'),
        dict(type='RandomFlip4TwoFrame', flip_ratio=0.5),
        dict(type='Resize4TwoFrame', img_scale=(640, 640), keep_ratio=True),
        dict(
            type='Pad4TwoFrames',
            pad_to_square=True,
            pad_val=dict(img=(114.0, 114.0, 114.0))),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='DefaultFormatBundle4TwoFrame'),
        dict(
            type='Collect4TwoFrame',
            keys=['img', 'img_2', 'img_2_gt', 'gt_bboxes', 'gt_labels'])
    ])
test_pipeline = [
    dict(type='LoadTwoImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize4TwoFrame', keep_ratio=True),
            dict(type='RandomFlip4TwoFrame'),
            dict(
                type='Pad4TwoFrames4TestMode',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='DefaultFormatBundle4TwoFrame'),
            dict(
                type='Collect4TwoFrameTestMode',
                keys=['img', 'img_2', 'gt_bboxes', 'gt_labels'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    persistent_workers=True,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDualSmokeDataset',
            ann_file='data/Data/csv/coco/annotations/instances_train2017.json',
            img_prefix='data/Data/csv/coco/images/train2017/',
            pipeline=[
                dict(type='LoadTwoImagesFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='MosaicTwoFrames', img_scale=(640, 640), pad_val=114.0),
            dict(
                type='RandomAffine4TwoFrames',
                scaling_ratio_range=(0.1, 2),
                border=(-320, -320)),
            dict(
                type='MixUp4TwoFrames',
                img_scale=(640, 640),
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(type='YOLOXHSVRandomAug4TwoFrames'),
            dict(type='RandomFlip4TwoFrame', flip_ratio=0.5),
            dict(
                type='Resize4TwoFrame', img_scale=(640, 640), keep_ratio=True),
            dict(
                type='Pad4TwoFrames',
                pad_to_square=True,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='DefaultFormatBundle4TwoFrame'),
            dict(
                type='Collect4TwoFrame',
                keys=['img', 'img_2', 'img_2_gt', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CocoDualSmokeDataset',
        ann_file='data/Data/csv/coco/annotations/instances_val2017.json',
        img_prefix='data/Data/csv/coco/images/val2017/',
        pipeline=[
            dict(type='LoadTwoImagesFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize4TwoFrame', keep_ratio=True),
                    dict(type='RandomFlip4TwoFrame'),
                    dict(
                        type='Pad4TwoFrames4TestMode',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle4TwoFrame'),
                    dict(
                        type='Collect4TwoFrameTestMode',
                        keys=['img', 'img_2', 'gt_bboxes', 'gt_labels'])
                ])
        ]),
    test=dict(
        type='CocoDualSmokeDataset',
        ann_file='data/Data/csv/coco/annotations/instances_val2017.json',
        img_prefix='data/Data/csv/coco/images/val2017/',
        pipeline=[
            dict(type='LoadTwoImagesFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize4TwoFrame', keep_ratio=True),
                    dict(type='RandomFlip4TwoFrame'),
                    dict(
                        type='Pad4TwoFrames4TestMode',
                        pad_to_square=True,
                        pad_val=dict(img=(114.0, 114.0, 114.0))),
                    dict(type='DefaultFormatBundle4TwoFrame'),
                    dict(
                        type='Collect4TwoFrameTestMode',
                        keys=['img', 'img_2', 'gt_bboxes', 'gt_labels'])
                ])
        ]))
max_epochs = 300
num_last_epochs = 15
interval = 10
evaluation = dict(
    save_best='auto', interval=10, dynamic_intervals=[(286, 1)], metric='bbox')
work_dir = './work_dirs/yolox_s_8x8_300e_coco'
auto_resume = False
gpu_ids = [0]
