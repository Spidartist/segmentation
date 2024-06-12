pretrained = "/mnt/quanhd/ijepa_endoscopy_pretrained/splitted_target_encoder_jepa-ep500.pth.tar"
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='IJEPASegmentor',
    backbone=dict(
        pretrained=pretrained,
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
        with_cp=True,  # set with_cp=True to save memory
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    ),
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
    Kvasir="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/Kvasir",
    CVC_ClinicDB="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/CVC-ClinicDB",
    CVC_ColonDB="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/CVC-ColonDB",
    CVC_T="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/CVC-300",
    ETIS_Larib="/mnt/tuyenld/data/endoscopy/public_dataset/TestDataset/ETIS-LaribPolypDB",
)
