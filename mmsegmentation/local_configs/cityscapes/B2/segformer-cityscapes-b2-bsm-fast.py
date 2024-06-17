_base_ = ['../../../configs/segformer/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py']

model = dict(
    backbone=dict(
        tome_cfg=[
            dict(kv_mode='bsm', kv_r=0.9, kv_sx=4, kv_sy=4),
            dict(kv_mode='bsm', kv_r=0.9, kv_sx=4, kv_sy=4),
            dict(q_mode='bsm', q_r=0.9, q_sx=4, q_sy=4),
            dict(q_mode='bsm', q_r=0.9, q_sx=4, q_sy=4)
        ]
    )
)

train_dataloader = dict(batch_size=4, num_workers=8)
