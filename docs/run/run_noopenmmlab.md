# Use the SegFormer++ without OpenMMLab:

## Building a Model

- Use [build_model.py](../../model_without_OpenMMLab/segformer_plusplus/build_model.py) to build preset and custom SegFormer++ models

Navigate to model_without_OpenMMLab.
```python
from segformer_plusplus.build_model import create_model
# backbone: choose from ['b0', 'b1', 'b2', 'b3', 'b4', 'b5']
# tome_strategy: choose from ['bsm_hq', 'bsm_fast', 'n2d_2x2']
out_channels = 19  # number of classes, e.g. 19 for cityscapes
model = create_model('b5', 'bsm_hq', out_channels=out_channels, pretrained=True)
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
checkpoint_path = "path_to_your_checkpoint.pth that you downloaded (links in Readme)"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
```
An Example can be found in [start_cityscape_benchmark.py](../../model_without_OpenMMLab/segformer_plusplus/start_cityscape_benchmark.py)

## Image Preperation

Images can be imported via PIL and then converted into RGB:

```python
from PIL import Image
image_path = "path_to_your_image.jpg"
image = Image.open(image_path).convert("RGB")
```

After that we can transform the image into a torch-Tensor and calculate Mean and STD:

```python
import torchvision.transforms as T
img_tensor = T.ToTensor()(image)
mean = [0.485, 0.456, 0.406]  # ImageNet mean
std = [0.229, 0.224, 0.225]  # ImageNet std
```

Now the Tensor of the Image can be transformed:

```python
device = 'cuda:0'
resolution = (1024, 1024)  # image_size as wished
transform = T.Compose([
T.Resize(resolution),
T.ToTensor(),
T.Normalize(mean=mean, std=std)
])
img_tensor = transform(image).unsqueeze(0).to(device)
```

```python
output = model(img_tensor).squeeze(0)
```

Visualize the results (this is for cityscapes classes):

```python
import numpy as np
segmentation_map = np.argmax(output.detach().cpu().numpy(), axis=0)
# Official Cityscapes colors for train IDs 0-18
cityscapes_colors = np.array([
    [128,  64, 128], # 0: road
    [244,  35, 232], # 1: sidewalk
    [ 70,  70,  70], # 2: building
    [102, 102, 156], # 3: wall
    [190, 153, 153], # 4: fence
    [153, 153, 153], # 5: pole
    [250, 170,  30], # 6: traffic light
    [220, 220,   0], # 7: traffic sign
    [107, 142,  35], # 8: vegetation
    [152, 251, 152], # 9: terrain
    [ 70, 130, 180], # 10: sky
    [220,  20,  60], # 11: person
    [255,   0,   0], # 12: rider
    [  0,   0, 142], # 13: car
    [  0,   0,  70], # 14: truck
    [  0,  60, 100], # 15: bus
    [  0,  80, 100], # 16: train
    [  0,   0, 230], # 17: motorcycle
    [119,  11,  32], # 18: bicycle
], dtype=np.uint8)

color_image = cityscapes_colors[segmentation_map]
```

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.imshow(color_image)
plt.title("Semantic Segmentation Visualization")
plt.axis('off')
plt.show()
```

> Note: You have to install matplotlib for visualization.


## Token-Merge Setting

For information to the settings for the Token Merging look [here](../../docs/run/token_merging.md).

