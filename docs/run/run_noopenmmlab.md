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
image_path = "path_to_your_image.png"
image = Image.open(image_path).convert("RGB")
```

After that, convert the image into a torch tensor:

```python
import torch
import numpy as np

img_tensor = torch.from_numpy(np.array(image) / 255.0)
img_tensor = img_tensor.permute(2, 0, 1).float().unsqueeze(0)  # (1, C, H, W)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_tensor = img_tensor.to(device)
```

Now we can load the model:

```python
from segformer_plusplus.build_model import create_model

out_channels = 19
model = create_model(
    backbone='b5', 
    tome_strategy='bsm_hq', 
    out_channels=out_channels, 
    pretrained=False
).to(device)

model.load_state_dict(torch.load("path_to_checkpoint", map_location=device))
model.eval()
```

Inference:
```python
with torch.no_grad():
    output = model(img_tensor)
    segmentation_map = torch.argmax(output, dim=1).squeeze().cpu().numpy()
```

Visualize the results (this is for cityscapes classes):

```python
import numpy as np

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

# Map each class to its corresponding color
height, width = segmentation_map.shape
color_image = np.zeros((height, width, 3), dtype=np.uint8)
for class_index in range(len(cityscapes_colors)):
    color_image[segmentation_map == class_index] = cityscapes_colors[class_index]
```

Display and save output:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(color_image)
plt.title("Semantic Segmentation Visualization")
plt.axis('off')
plt.show()
# Save the colorized output as an image - important when using a System without GUI
plt.imsave("segmentation_output.png", color_image)
```

> Note: You have to install matplotlib for visualization.


## Token-Merge Setting

For information to the settings for the Token Merging look [here](../../docs/run/token_merging.md).

