import torch
from PIL import Image
import torchvision.transforms as T
import os
from typing import Union, List, Tuple
import numpy as np

from .utils.benchmark import benchmark


# Gerät auswählen
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

def cityscape_benchmark(
        model: torch.nn.Module,
        image_path: str,
        batch_size: Union[int, List[int]] = 1,
        image_size: Union[Tuple[int], List[Tuple[int]]] = (3, 1024, 1024),
        save_output: bool = True,

):
    """
    Calculate the FPS of a given model using an actual Cityscapes image.

    Args:
        model: instance of a model (e.g. SegFormer)
        image_path: the path to the Cityscapes image
        batch_size: the batch size(s) at which to calculate the FPS (e.g. 1 or [1, 2, 4])
        image_size: the size of the images to use (e.g. (3, 1024, 1024))
        save_output: whether to save the output prediction (default True)

    Returns:
        the FPS values calculated for all image sizes and batch sizes in the form of a dictionary
    """


    if isinstance(batch_size, int):
        batch_size = [batch_size]
    if isinstance(image_size, tuple):
        image_size = [image_size]

    values = {}
    throughput_values = []

    model = model.to(device)
    model.eval()

    assert os.path.exists(image_path), f"Image not found: {image_path}"
    image = Image.open(image_path).convert("RGB")

    img_tensor = T.ToTensor()(image)
    mean = img_tensor.mean(dim=(1, 2))
    std = img_tensor.std(dim=(1, 2))
    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")

    transform = T.Compose([
        T.Resize((image_size[0][1], image_size[0][2])),
        T.ToTensor(),
        T.Normalize(mean=mean.tolist(),
                    std=std.tolist())
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

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

    if save_output:
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

            # Speichere Prediction als Text ab
            cwd = os.getcwd()
            output_path = os.path.join(cwd, 'segformer_plusplus', 'cityscapes_prediction_output.txt')
            np.savetxt(output_path, pred, fmt="%d")

            print("Prediction saved as cityscapes_prediction_output.txt")

    return values
