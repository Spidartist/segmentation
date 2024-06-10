norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[117.9630,  79.7130,  63.8520],
    std=[81.4725, 63.9030, 56.2275],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_jepa-ep500.pth.tar', # noqa
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
        qkv_bias=True
    ),
    decode_head=dict(
        type='DPTHead',
        in_channels=(768, 768, 768, 768),
        channels=256,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        num_classes=150,
        readout_type='project',
        input_transform='multiple_select',
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))  # yapf: disable
