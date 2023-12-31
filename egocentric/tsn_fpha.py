
#dataset
dataset_type = 'VideoDataset'
data_root = 'data/video_fpha/train'
data_root_val = 'data/video_fpha/val'
ann_file_train = 'data/video_fpha/fpha_train.txt'
ann_file_val = 'data/video_fpha/fpha_val.txt'
auto_scale_lr = dict(base_batch_size=256, enable=False)

#hooks
default_hooks = dict(
    checkpoint=dict(
        interval=3, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))

#env
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = './checkpoints/tsn_fpha.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
#model
model = dict(
    backbone=dict(
        depth=50,
        norm_eval=False,
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',
        type='ResNet'),
    cls_head=dict(
        average_clips='prob',
        consensus=dict(dim=1, type='AvgConsensus'),
        dropout_ratio=0.4,
        in_channels=2048,
        init_std=0.01,
        num_classes=45, #classes
        spatial_type='avg',
        type='TSNHead'),
    data_preprocessor=dict(
        format_shape='NCHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    test_cfg=None,
    train_cfg=None,
    type='Recognizer2D')
#optim
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=100,
        gamma=0.1,
        milestones=[
            40,
            80,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = False
#test
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/video_fpha/fpha_val.txt',
        data_prefix=dict(video='data/video_fpha/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=25,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='TenCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='AccMetric')
test_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='TenCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
#train
train_cfg = dict( #max_epochs
    max_epochs=100, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='data/video_fpha/fpha_train.txt',
        data_prefix=dict(video='data/video_fpha/train'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1, frame_interval=1, num_clips=3,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(
                input_size=224,
                max_wh_scale_gap=1,
                random_crop=False,
                scales=(
                    1,
                    0.875,
                    0.75,
                    0.66,
                ),
                type='MultiScaleCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=1, frame_interval=1, num_clips=3, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(
        input_size=224,
        max_wh_scale_gap=1,
        random_crop=False,
        scales=(
            1,
            0.875,
            0.75,
            0.66,
        ),
        type='MultiScaleCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
#val
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file='data/video_fpha/fpha_val.txt',
        data_prefix=dict(video='data/video_fpha/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=1,
                frame_interval=1,
                num_clips=3,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        test_mode=True,
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='AccMetric')
val_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/tsn_fpha'
