# Using the SegFormer++ Model via PyTorch Hub

This document explains how to use a pre-trained SegFormer++ model and its associated data transformations by loading them directly from a GitHub repository using PyTorch Hub. The process streamlines model access, making it easy to integrate the model into your projects with a simple one-liner.

## Prerequisites

Before running the script, ensure you have PyTorch installed. You also need to install the following dependencies, which are required by the model and its entry points:

```bash
pip install tomesd omegaconf numpy rich yapf addict tqdm packaging torchvision
```
## How It Works

The provided Python script demonstrates a full workflow, from loading the model and transformations to running inference on a dummy image.

## Step 1: Loading the Model

You can easily load the model from torchhub.
The parameters are:
- `pretrained`: If set to True, it loads the model with pre-trained ImageNet weights.
- `backbone`: Specifies the backbone architecture (e.g., 'b5' for MiT-B5). Other options include 'b0', 'b1', 'b2', 'b3', and 'b4'.
- `tome_strategy`: Defines the token merging strategy. Options include 'bsm_hq' (high quality), 'bsm_fast' (faster), and 'n2d_2x2' (non-overlapping 2x2).
- `checkpoint_url`: A URL to a specific checkpoint file. This way you can load our trained model weights that you can find in the README. Make sure, that your weight fit to the model size and number of classes.
- `out_channels`: The number of output classes for segmentation (e.g., 19 for Cityscapes).

```python
import torch
model = torch.hub.load(
    'KieDani/SegformerPlusPlus',
    'segformer_plusplus',
    pretrained=True,
    backbone='b5',
    tome_strategy='bsm_hq',
    checkpoint_url='https://mediastore.rz.uni-augsburg.de/get/yzE65lzm6N/',  # URL to checkpoints, optional
    out_channels=19,
)
model.eval()  # Set the model to evaluation mode
```

## Step 2: Loading Data Transformations

The data_transforms entry point returns a torchvision.transforms.Compose object, which encapsulates the standard preprocessing steps required by the model (resizing and normalization).

```python
# Load the data transformations
transform = torch.hub.load(
    'KieDani/SegformerPlusPlus',
    'data_transforms',
)
```

## Step 3: Preparing the Image and Running Inference

After loading the model and transformations, you can apply them to an input image. The script creates a dummy image for this example, but in a real-world scenario, you would load an image from your file system.

```python
from PIL import Image

# In a real-world scenario, you would load your image here:
# image = Image.open('path_to_your_image.jpg').convert('RGB')
dummy_image = Image.new('RGB', (1300, 1300), color='red')

# Apply the transformations
input_tensor = transform(dummy_image).unsqueeze(0)  # Add a batch dimension

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Process the output tensor to get the final segmentation map
segmentation_map = torch.argmax(output.squeeze(0), dim=0)
```

The final segmentation_map is a tensor where each pixel value represents the predicted class (from 0 to 18).

## Full Script

Below is the complete, runnable script for your reference.

```python
import torch.hub
from PIL import Image

# --- IMPORTANT: TorchHub Dependencies ---
# Install the dependencies via:
# pip install tomesd omegaconf numpy rich yapf addict tqdm packaging torchvision

# Load the SegFormer++ model with predefined parameters.
print("Loading SegFormer++ Model...")
# Replace 'your_username/your_repo' with the actual path to your repository
model = torch.hub.load(
    'KieDani/SegformerPlusPlus',  # This is a placeholder, replace it with your actual GitHub path
    'segformer_plusplus',
    pretrained=True,
    backbone='b5',
    tome_strategy='bsm_hq',
    checkpoint_url='https://mediastore.rz.uni-augsburg.de/get/yzE65lzm6N/',
    out_channels=19,
)
model.eval()
print("Model loaded successfully.")

# Load the data transformations via the 'data_transforms' entry point.
print("Loading data transformations...")
transform = torch.hub.load(
    'KieDani/SegformerPlusPlus',  # Placeholder, replace it with your actual GitHub path
    'data_transforms',
)
print("Transformations loaded successfully.")

# --- Example for Image Preparation and Inference ---
# Create a dummy image, as we don't need a real image file.
# In a real scenario, you would load an image from the hard drive, e.g.
# from PIL import Image
# image = Image.open('path_to_your_image.jpg').convert('RGB')
print("Creating a dummy image for demonstration...")
dummy_image = Image.new('RGB', (1300, 1300), color='red')
print("Original image size:", dummy_image.size)

# Apply the transformations loaded from the Hub to the image.
print("Applying transformations to the image...")
input_tensor = transform(dummy_image).unsqueeze(0)  # Adds a batch dimension
print("Transformed image tensor size:", input_tensor.shape)

# Run inference.
print("Running inference...")
with torch.no_grad():
    output = model(input_tensor)

# The output tensor has the shape [1, num_classes, height, width]
# We remove the batch dimension (1)
output_tensor = output.squeeze(0)

print(f"\nInference completed. Output tensor size: {output_tensor.shape}")

# To get the final segmentation map, you would use argmax.
segmentation_map = torch.argmax(output_tensor, dim=0)
print(f"Size of the generated segmentation map: {segmentation_map.shape}")
```
