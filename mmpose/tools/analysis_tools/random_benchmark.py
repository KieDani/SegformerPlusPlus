import argparse

import numpy as np
from mmengine import Config

import torch

from mmpose.apis import init_model
from tools.analysis_tools import tome_benchmark


def parse_image_size(size_str):
    try:
        # Split the string by 'x'
        dimensions = size_str.split('x')
        # Convert each part to an integer
        if len(dimensions) != 3:
            raise ValueError("Size must be in the format 'CxHxW'")
        dimensions = tuple(map(int, dimensions))
        return dimensions
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark model using random tensors')
    parser.add_argument('-c', '--configs', required=True, type=str, nargs='+')
    parser.add_argument('-i', '--im_size', default=(3, 480, 640), type=parse_image_size, nargs='+')
    parser.add_argument('-b', '--batch_size', default=8, type=int, nargs='+')
    args = parser.parse_args()
    return args


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args = parse_args()
cfgs = args.configs
batch_sizes = args.batch_size
image_sizes = args.im_size

if isinstance(batch_sizes, int):
    batch_sizes = [batch_sizes]
if isinstance(image_sizes, tuple):
    image_sizes = [image_sizes]

values = {}
throughput_values = []

for cfg in cfgs:
    print(cfg)
    x = cfg
    cfg = Config.fromfile(cfg)
    model = init_model(cfg).to(device)
    for i in image_sizes:
        # fill with fps for each batch size
        fps = []
        for b in batch_sizes:
            for _ in range(4):
                # Baseline benchmark
                if i[1] >= 1024:
                    r = 16
                else:
                    r = 32
                baseline_throughput = tome_benchmark.benchmark(
                    model,
                    device=device,
                    verbose=True,
                    runs=r,
                    batch_size=b,
                    input_size=i
                )
                throughput_values.append(baseline_throughput)
            throughput_values = np.asarray(throughput_values)
            throughput = np.around(np.mean(throughput_values), decimals=2)
            print(x, 'Im_size:', i, 'Batch_size:', b, 'Mean:', throughput, 'Std:',
                  np.around(np.std(throughput_values), decimals=2))
            throughput_values = []
            fps.append({b: throughput})
        key = x + ' ' + str(i)
        values[key] = fps

print(values)
