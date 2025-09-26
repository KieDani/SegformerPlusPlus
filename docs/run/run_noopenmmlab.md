# Use the SegFormer++ without OpenMMLab:

## Building a Model

- Use [build_model.py](../../model_without_OpenMMLab/segformer_plusplus/build_model.py) to build preset and custom SegFormer++ models

Navigate to model_without_OpenMMLab.
```python
from segformer_plusplus.build_model import create_model
# backbone: choose from ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
# head: choose from ['bsm_hq', 'bsm_fast', 'n2d_2x2']
model = create_model('b5', 'bsm_hq', pretrained=True)
```
Running this code snippet yields our SegFormer++<sub>HQ</sub> model pretrained on ImageNet.

- Use [random_benchmark.py](../../model_without_OpenMMLab/segformer_plusplus/random_benchmark.py) to evaluate a model in terms of FPS

```python
from segformer_plusplus.random_benchmark import random_benchmark
v = random_benchmark(model)
```
Calculate the FPS using our script.

## Loading a Checkpoint

[Checkpoints](../../README.md) are provided in this Repository.
They can be loaded and integrated into the model via PyTorch:
```python
import torch
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
```
An Example can be found in [start_cityscape_benchmark.py](../../model_without_OpenMMLab/segformer_plusplus/start_cityscape_benchmark.py)

## Image Preperation

Images can be imported via PIL and then converted into RGB:

```python
from PIL import Image
image = Image.open(image-path).convert("RGB")
```

After that we can transform the image into a torch-Tensor and calculate Mean and STD:

```python
import torchvision.transforms as T
img_tensor = T.ToTensor()(image)
mean = img_tensor.mean(dim=(1, 2))
std = img_tensor.std(dim=(1, 2))
```

Now the Tensor of the Image can be transformed:

```python
transform = T.Compose([
        T.Resize((image_size[0][1], image_size[0][2])), #image_size as wished
        T.ToTensor(),
        T.Normalize(mean=mean.tolist(),
                    std=std.tolist())
    ])
img_tensor = transform(image).unsqueeze(0).to(device)
```

> **Note:** The provided checkpoints were trained with normalization of the dataset. 
> For precise results, use the dataset's mean and standard deviation instead of calculating per-image values.

```python
output = model(img_tensor)
```

## Token-Merge Setting

For information to the settings for the Token Merging look [here](../../docs/run/token_merging.md).

