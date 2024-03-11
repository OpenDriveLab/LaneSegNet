_base_ = []
custom_imports = dict(imports=['projects.bevformer', 'projects.lanesegnet'])

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -25.6, -2.3, 51.2, 25.6, 1.7]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ['lane_segment', 'ped_crossing']
class_nums = len(class_names)

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
num_cams = 7
pts_dim = 3

dataset_type = 'OpenLaneV2_subset_A_LaneSegNet_Dataset'
data_root = 'data/OpenLane-V2/'

para_method = 'fix_pts_interp'
method_para = dict(n_points=10)
code_size = 3 * method_para['n_points'] * pts_dim

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_ffn_cfg_ = dict(
    type='FFN',
    embed_dims=_dim_,
    feedforward_channels=_ffn_dim_,
    num_fcs=2,
    ffn_drop=0.1,
    act_cfg=dict(type='ReLU', inplace=True),
),

_num_levels_ = 4
bev_h_ = 100
bev_w_ = 200

model = dict(
    type='LaneSegNet',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    bev_constructor=dict(
        type='BEVFormerConstructer',
        num_feature_levels=_num_levels_,
        num_cams=num_cams,
        embed_dims=_dim_,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        pc_range=point_cloud_range,
        bev_h=bev_h_,
        bev_w=bev_w_,
        rotate_center=[bev_h_//2, bev_w_//2],
        encoder=dict(
            type='BEVFormerEncoder',
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            transformerlayers=dict(
                type='BEVFormerLayer',
                attn_cfgs=[
                    dict(
                        type='TemporalSelfAttention',
                        embed_dims=_dim_,
                        num_levels=1),
                    dict(
                        type='SpatialCrossAttention',
                        embed_dims=_dim_,
                        num_cams=num_cams,
                        pc_range=point_cloud_range,
                        deformable_attention=dict(
                            type='MSDeformableAttention3D',
                            embed_dims=_dim_,
                            num_points=8,
                            num_levels=_num_levels_)
                    )
                ],
                ffn_cfgs=_ffn_cfg_,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm'))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
    ),
    lane_head=dict(
        type='LaneSegHead',
        num_classes=class_nums,
        num_lane_type_classes=3,
        in_channels=_dim_,
        num_query=200,
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        pts_dim=pts_dim,
        sync_cls_avg_factor=False,
        with_box_refine=True,
        code_size=code_size,
        code_weights= [1.0 for i in range(code_size)],
        transformer=dict(
            type='LaneSegNetTransformer',
            embed_dims=_dim_,
            points_num=method_para['n_points'],
            pts_dim=pts_dim,
            decoder=dict(
                type='LaneSegNetDecoder',
                num_layers=6,
                return_intermediate=True,
                pc_range=point_cloud_range,
                pts_dim=pts_dim,
                transformerlayers=dict(
                    type='CustomDetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='LaneAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            num_points=32,
                            num_levels=1),
                    ],
                    ffn_cfgs=_ffn_cfg_,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(type='LaneSegmentPseudoCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.025),
        loss_lane_type=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=0.1),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=3.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=3.0)),
    lclc_head=dict(
        type='RelationshipHead',
        in_channels_o1=_dim_,
        in_channels_o2=_dim_,
        shared_param=False,
        loss_rel=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=5)),
    # model training and testing settings
    train_cfg=dict(
        lane=dict(
            assigner=dict(
                type='LaneSegmentHungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=1.5),
                reg_cost=dict(type='LaneL1Cost', weight=0.025),
                mask_cost=dict(type='CrossEntropyLossCost', weight=3.0, use_sigmoid=True),
                dice_cost=dict(type='DiceCost', weight=3.0, pred_act=False, eps=1.0),
                pc_range=point_cloud_range))))

train_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3DLaneSegment',
         with_lane_3d=True, with_lane_label_3d=True, with_lane_adj=True, with_lane_type=True,
         with_bbox=False, with_label=False, with_lane_lste_adj=False),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='GridMaskMultiViewImage'),
    dict(type='LaneSegmentParameterize3D', method=para_method, method_para=method_para),
    dict(type='GenerateLaneSegmentMask', points_num=method_para['n_points'], bev_h=bev_h_, bev_w=bev_w_),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=[
        'img', 'gt_lanes_3d', 'gt_lane_labels_3d', 'gt_lane_adj',
        'gt_instance_masks', 'gt_lane_left_type', 'gt_lane_right_type'])
]

test_pipeline = [
    dict(type='CustomLoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CropFrontViewImageForAv2'),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImageSame2Max', size_divisor=32),
    dict(type='CustomFormatBundle3DLane', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img'])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_train_lanesegnet.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        split='train',
        filter_map_change=True,
        points_num=method_para['n_points'],
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val_lanesegnet.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        points_num=method_para['n_points'],
        test_mode=True),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'data_dict_subset_A_val_lanesegnet.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        split='val',
        points_num=method_para['n_points'],
        test_mode=True)
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
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
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=24, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1, max_keep_ckpts=1)

dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)
