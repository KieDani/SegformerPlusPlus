_base_ = ['../../../configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py']

model = dict(
    backbone=dict(
        tome_cfg=[
            dict(q_mode='n2d', q_s=(2, 2)),
            dict(q_mode='n2d', q_s=(2, 2)),
            dict(q_mode='n2d', q_s=(2, 2)),
            dict(q_mode='n2d', q_s=(2, 2))
        ]
    )
)

train_dataloader = dict(batch_size=4, num_workers=8)
