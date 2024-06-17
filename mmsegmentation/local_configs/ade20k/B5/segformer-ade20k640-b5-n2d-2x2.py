_base_ = ['../../../configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py']

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

train_dataloader = dict(batch_size=8, num_workers=8)
