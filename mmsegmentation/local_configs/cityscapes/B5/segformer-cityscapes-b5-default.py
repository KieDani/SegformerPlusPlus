_base_ = ['../../../configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py']

train_dataloader = dict(batch_size=4, num_workers=8)