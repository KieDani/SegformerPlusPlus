_base_ = ['./segformer_mit_b1.py']

backbone = dict(
        embed_dims=64,
        num_layers=[3, 4, 6, 3]
)
