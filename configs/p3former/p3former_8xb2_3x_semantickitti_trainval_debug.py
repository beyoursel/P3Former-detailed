_base_ = [
    '../_base_/datasets/semantickitti_panoptic_lpmix.py', '../_base_/models/p3former.py',
    '../_base_/default_runtime.py'
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

grid_shape = [160, 120, 12]
model = dict(
    data_preprocessor=dict(
        type='_Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            max_num_points=20000,
            max_voxels=-1,
        ),
    ),
    voxel_encoder=dict(
        feat_channels=[8, 16],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=8,
        return_point_feats=False),
    backbone=dict(
        grid_size=grid_shape,
        input_channels=8,
        base_channels=16,
        more_conv=True,
        out_channels=16),
    decode_head=dict(
        num_decoder_layers=1,
        num_queries=16,
        embed_dims=16,
        grid_size=grid_shape,
        cls_channels=(16, 16, 20),
        mask_channels=(16, 16, 16, 16, 16),
        thing_class=[0,1,2,3,4,5,6,7],
        stuff_class=[8,9,10,11,12,13,14,15,16,17,18],
        ignore_index=19
    ))


lr = 0.0008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.2)
]

train_dataloader = dict(
    batch_size=1, 
    num_workers=1,
    dataset=dict(dataset=dict(ann_file='semantickitti_infos_trainval.pkl'))
)

test_dataloader = dict(
    batch_size=1, 
    num_workers=1,
)

val_dataloader = test_dataloader

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True,
    ),
    visualization=dict(type='Det3DVisualizationHook', draw=False)       
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head',
        'p3former.segmentors.p3former',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'datasets.semantickitti_dataset',
        'datasets.transforms.loading',
        'datasets.transforms.transforms_3d',
    ],
    allow_failed_imports=False)
