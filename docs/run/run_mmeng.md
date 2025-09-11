# Use the SegFormer++ outside MMSegmentation/MMPose

## How to build and use a Model

- Use [build_model.py](../../model/segformer/build_model.py) to build preset and custom SegFormer++ models

```python
model = create_model('b5', 'bsm_hq', pretrained=True)
```
Running this code snippet yields our SegFormer++<sub>HQ</sub> model pretrained on ImageNet.

- Use [random_benchmark.py](../../model/segformer/random_benchmark.py) to evaluate a model in terms of FPS

```python
v = random_benchmark(model)
```
Calculate the FPS using our script.

## How to execute the benchmark via scripts

- Use [start_random_benchmark.py](../../model/segformer/start_random_benchmark.py) to evaluate a model with random input in terms of FPS:

```bash
python3 -m segformer_plusplus.start_random_benchmark
```
  
- Use [start_cityscape_benchmark.py](../../model/segformer/start_cityscape_benchmark.py) to evaluate a model with a cityscape-picture as input in terms of FPS and output:

```bash
python3 -m segformer_plusplus.start_cityscape_benchmark --backbone <BACKBONE_VERSION> --head <HEAD_TYPE> --checkpoint <PATH_TO_CHECKPOINT>
```

The parameters can take the following values:

--backbone: Model backbone version. Possible values: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'. Default is 'b5'.

--head: Model head type. Possible values: 'bsm_hq', 'bsm_fast', 'n2d_2x2'. Default is 'bsm_hq'.

--checkpoint: Path to a .pth checkpoint file (optional). If not provided, the script may use the model pretrained on ImageNet.

For example it can look like this:

```bash
python3 -m segformer_plusplus.start_cityscape_benchmark --backbone b5 --head bsm_hq --checkpoint segformer_b5_cityscapes_hq.pth
```

