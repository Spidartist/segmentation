pretrained = "/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_jepa-ep500.pth.tar"
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='IJEPASegmentor',
    pretrained=pretrained,
    backbone=dict(
        type='IJEPA',
        img_size=[256],
        patch_size=16,
        in_channels=3,
        use_gap=False,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained)
    ),
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=1, # 1 for binary classification
        norm_cfg=norm_cfg,
        align_corners=False,
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

test_path = dict(
    Kvasir="/home/s/tuyenld/DATA/public_dataset/TestDataset/Kvasir",
    CVC_ClinicDB="/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ClinicDB",
    CVC_ColonDB="/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-ColonDB",
    CVC_T="/home/s/tuyenld/DATA/public_dataset/TestDataset/CVC-300",
    ETIS_Larib="/home/s/tuyenld/DATA/public_dataset/TestDataset/ETIS-LaribPolypDB",
)
