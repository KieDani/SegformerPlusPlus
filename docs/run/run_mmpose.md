# Run Experiments (MMPose)

The configuration files for our models can be found here:

- [Custom configuration files](../../mmpose/local_configs)

## Training

```shell
python tools/train.py 'path/to/config'
```

## Evaluation

```shell
python tools/test.py 'path/to/config' 'path/to/checkpoint'
```

## FPS

```shell
python tools/analysis_tools/random_benchmark.py -c 'path/to/config' -i 3x480x640 -b 16
```