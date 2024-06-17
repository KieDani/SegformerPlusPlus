_base_ = ['./segformer_mit_b0.py']

backbone = dict(
        embed_dims=64,
)
decode_head = dict(
    in_channels=[64, 128, 320, 512]
)
