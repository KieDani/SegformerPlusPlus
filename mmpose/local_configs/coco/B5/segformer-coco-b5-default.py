_base_ = ['../td-hm_segformer-b0-210e_coco-384x288.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3]
    ),
    head=dict(in_channels=[64, 128, 320, 512])
)
