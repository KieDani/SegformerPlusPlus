_base_ = ['../tm-hm_segformer-b0-210e_jump-640x480.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 6, 40, 3],
        tome_cfg=[
            dict(q_mode='n2d', q_s=(2, 2)),
            dict(q_mode='n2d', q_s=(2, 2)),
            dict(q_mode='n2d', q_s=(2, 2)),
            dict(q_mode='n2d', q_s=(2, 2))
        ]
    ),
    head=dict(in_channels=[64, 128, 320, 512])
)
