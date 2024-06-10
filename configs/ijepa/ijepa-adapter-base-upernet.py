_base_ = [
    '../_base_/models/upernet_ijepa.py', '../_base_/datasets/public_dataset.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
custom_imports = dict(imports=["mmseg_custom"])

# crop_size = (352, 352)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_jepa-ep500.pth.tar', # noqa
    backbone=dict(
        type='IJEPAAdapter',
        img_size=[256],
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        init_values=1e-6,
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=12,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    ),
    neck=dict(embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        in_channels=[768, 768, 768, 768], 
        num_classes=2, 
        # channels=768,
        out_channels=2,
        # threshold=0.5
    ),
    auxiliary_head=dict(
        in_channels=768, 
        num_classes=2,
        out_channels=2,
        # threshold=0.5
    ),
)
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(341, 341)))

optimizer = dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.65),
    constructor='LayerDecayOptimizerConstructor',
    accumulative_counts=16)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

# mixed precision
fp16 = dict(loss_scale='dynamic')

train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=2)
# test_cfg = None
test_dataloader = val_dataloader    
# work_dir = './work_dirs/tutorial'