import torch.hub
from PIL import Image

# --- IMPORTANT: TorchHub Dependencies ---
# Install the dependencies via:
# pip install tomesd omegaconf numpy rich yapf addict tqdm packaging torchvision

# Load the SegFormer++ model with predefined parameters.
print("Loading SegFormer++ Model...")
# Replace 'your_username/your_repo' with the actual path to your repository
model = torch.hub.load(
    'KieDani/SegformerPlusPlus',
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
    'KieDani/SegformerPlusPlus',
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
