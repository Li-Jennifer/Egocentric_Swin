
#dataset
dataset_type = 'VideoDataset'
data_root = 'data/video_fpha/train'
data_root_val = 'data/video_fpha/val'
ann_file_train = 'data/video_fpha/fpha_train.txt'
ann_file_val = 'data/video_fpha/fpha_val.txt'
ann_file_test = 'data/video_fpha/fpha_val.txt'
auto_scale_lr = dict(base_batch_size=64, enable=False)
num_classes = 45

#hooks
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=3, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=True, interval=500,type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
#enc
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(io_backend='disk')
launcher = 'none'
load_from = './checkpoints/swin_small.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=30)

# model
model = dict(
    backbone=dict(
        arch='small',
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        drop_rate=0.0,
        mlp_ratio=4.0,
        patch_norm=True,
        patch_size=(
            2,
            4,
            4,
        ),
        pretrained=
        'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin_small_patch4_window7_224.pth',
        pretrained2d=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer3D',
        window_size=(
            8,
            7,
            7,
        )),
    cls_head=dict(
        average_clips='prob',
        dropout_ratio=0.5,
        in_channels=768,
        num_classes=45, # classes
        spatial_type='avg',
        type='I3DHead'),
    data_preprocessor=dict(
        format_shape='NCTHW',
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
    type='Recognizer3D')
#optim
optim_wrapper = dict(
    constructor='SwinOptimWrapperConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.02),
    paramwise_cfg=dict(
        absolute_pos_embed=dict(decay_mult=0.0),
        backbone=dict(lr_mult=0.1),
        norm=dict(decay_mult=0.0),
        relative_position_bias_table=dict(decay_mult=0.0)),
    type='AmpOptimWrapper')
#param
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=2.5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=30,
        begin=0,
        by_epoch=True,
        end=30,
        eta_min=0,
        type='CosineAnnealingLR'),
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
                clip_len=32,
                frame_interval=2,
                num_clips=4,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                224,
            ), type='Resize'),
            dict(crop_size=224, type='ThreeCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
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
        clip_len=32,
        frame_interval=2,
        num_clips=4,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        224,
    ), type='Resize'),
    dict(crop_size=224, type='ThreeCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
#train
train_cfg = dict( # max epochs
    max_epochs=64, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='data/video_fpha/fpha_train.txt',
        data_prefix=dict(video='data/video_fpha/train'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(type='RandomResizedCrop'),
            dict(keep_ratio=False, scale=(
                224,
                224,
            ), type='Resize'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        type='VideoDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=32, frame_interval=2, num_clips=1, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(type='RandomResizedCrop'),
    dict(keep_ratio=False, scale=(
        224,
        224,
    ), type='Resize'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
#val
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='data/video_fpha/fpha_val.txt',
        data_prefix=dict(video='data/video_fpha/val'),
        pipeline=[
            dict(io_backend='disk', type='DecordInit'),
            dict(
                clip_len=32,
                frame_interval=2,
                num_clips=1,
                test_mode=True,
                type='SampleFrames'),
            dict(type='DecordDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(crop_size=224, type='CenterCrop'),
            dict(input_format='NCTHW', type='FormatShape'),
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
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True,
        type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/swin_small'
