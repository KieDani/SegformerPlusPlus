import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from segformer_plusplus import create_model
import torch.nn.functional as F

# Cityscapes demo

model = create_model(
    backbone='b5',
    tome_strategy='bsm_hq',
    out_channels=19,
    pretrained=False
).cuda()

model.load_state_dict(torch.load('path/to/weights'))

image = cv2.imread('path/to/image.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0
image = np.transpose(image, (2, 0, 1))
image = torch.from_numpy(image).float()
image = image[None, ...]
image = image.cuda()

model.eval()
with torch.no_grad():
    x = model(image)
    x = F.softmax(x, dim=1)
    x = torch.argmax(x, dim=1).squeeze().cpu().numpy()

cityscapes_colormap = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
])

# Create an empty image with the same height and width as the predictions
height, width = x.shape
colorized_output = np.zeros((height, width, 3), dtype=np.uint8)

# Map each class to its corresponding color
for class_index in range(len(cityscapes_colormap)):
    colorized_output[x == class_index] = cityscapes_colormap[class_index]

# Display the colorized image
plt.imshow(colorized_output)
plt.axis('off')  # Hide axis
plt.show()

# Save the colorized output as an image (optional)
plt.imsave('colorized_output.png', colorized_output)
