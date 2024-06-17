_base_ = ['../../../configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py']

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

train_dataloader = dict(batch_size=8, num_workers=8)