_base_ = [
    '../_base_/models/dpt_ijepa.py', '../_base_/datasets/public_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

custom_imports = dict(imports=["mmseg_custom"])

pretrained = "/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_jepa-ep500.pth.tar"
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='IJEPA',
        img_size=[256],
        patch_size=16,
        in_channels=3,
        use_gap=True,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained)
    ),
    # neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], 
        num_classes=2, # 2 for binary classification
        out_channels=1, # 1 for binary classification
        threshold=0.5, # threshold for binary classification
        loss_decode=[
            # dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            # dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0),
            # dict(type='FocalLoss', use_sigmoid=True, loss_weight=1.0),
            dict(type="StructureLoss", loss_type="focal", loss_weight=1.0)
        ],
    ),
    # auxiliary_head=None,
    # auxiliary_head=dict(
    #     in_channels=768, 
    #     num_classes=2,  # 2 for binary classification
    #     out_channels=1, # 1 for binary classification
    #     threshold=0.3, # threshold for binary classification
    #     loss_decode=[
    #         # dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
    #         # dict(type='DiceLoss', use_sigmoid=True, loss_weight=0.4),
    #         # dict(type='FocalLoss', use_sigmoid=True, loss_weight=0.4),
    #         dict(type="StructureLoss", loss_type="focal", loss_weight=0.4)
    #     ]
    # ),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)),
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
        }),
    # paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95),
    # constructor='LayerDecayOptimizerConstructor_Custom',
    # accumulative_counts=16
)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
test_dataloader = val_dataloader
