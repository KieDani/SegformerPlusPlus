_base_ = ['../../../configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py']

train_dataloader = dict(batch_size=8, num_workers=8)
