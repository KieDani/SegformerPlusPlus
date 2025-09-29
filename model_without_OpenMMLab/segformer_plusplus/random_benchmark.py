from typing import Union, List, Tuple

import numpy as np
import torch

from .utils import benchmark

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def random_benchmark(
        model: torch.nn.Module,
        batch_size: Union[int, List[int]] = 1,
        image_size: Union[Tuple[int], List[Tuple[int]]] = (3, 1024, 1024),
):
    """
    Calculate the FPS of a given model using randomly generated tensors.

    Args:
        model: instance of a model (e.g. SegFormer)
        batch_size: the batch size(s) at which to calculate the FPS (e.g. 1 or [1, 2, 4])
        image_size: the size of the images to use (e.g. (3, 1024, 1024))

    Returns: the FPS values calculated for all image sizes and batch sizes in the form of a dictionary

    """
    if isinstance(batch_size, int):
        batch_size = [batch_size]
    if isinstance(image_size, tuple):
        image_size = [image_size]

    values = {}
    throughput_values = []

    for i in image_size:
        # fill with fps for each batch size
        fps = []
        for b in batch_size:
            for _ in range(4):
                # Baseline benchmark
                if i[1] >= 1024:
                    r = 16
                else:
                    r = 32
                baseline_throughput = benchmark(
                    model.to(device),
                    device=device,
                    verbose=True,
                    runs=r,
                    batch_size=b,
                    input_size=i
                )
                throughput_values.append(baseline_throughput)
            throughput_values = np.asarray(throughput_values)
            throughput = np.around(np.mean(throughput_values), decimals=2)
            print('Im_size:', i, 'Batch_size:', b, 'Mean:', throughput, 'Std:',
                  np.around(np.std(throughput_values), decimals=2))
            throughput_values = []
            fps.append({b: throughput})
        values[i] = fps
    return values
