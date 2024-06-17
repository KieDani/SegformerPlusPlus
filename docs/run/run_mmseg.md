# Run Experiments (MMSegmentation)

The configuration files for our models can be found here:

- [Custom configuration files](../../mmsegmentation/local_configs)

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
python tools/analysis_tools/random_benchmark.py -c 'path/to/config' -i 3x1024x1024 -b 8
```