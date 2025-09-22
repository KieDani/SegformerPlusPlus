norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone = dict(
    type='MixVisionTransformer',
    in_channels=3,
    embed_dims=32,
    num_stages=4,
    num_layers=[2, 2, 2, 2],
    num_heads=[1, 2, 5, 8],
    patch_sizes=[7, 3, 3, 3],
    sr_ratios=[8, 4, 2, 1],
    out_indices=(0, 1, 2, 3),
    mlp_ratio=4,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1
)
decode_head = dict(
    type='SegformerHead',
    in_channels=[32, 64, 160, 256],
    in_index=[0, 1, 2, 3],
    channels=256,
    dropout_ratio=0.1,
    out_channels=19,
    norm_cfg=norm_cfg,
    align_corners=False,
    interpolate_mode='bilinear'
)
