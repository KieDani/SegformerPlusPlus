# Use the SegFormer++ outside MMSegmentation/MMPose

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
