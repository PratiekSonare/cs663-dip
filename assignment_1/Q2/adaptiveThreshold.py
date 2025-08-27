#1. convert all images to unit8 dtype, i.e, (0, 255) range of values
#2. average out the RGB channels for grayscale
#3. apply adaptive niblack's thresholding method - which takes in mean and variance of local window to calculate local threshold

# some input images are float with max() <= 1 and some are already uint8;

# input_1 range:  {np.float32(0.0), np.float32(1.0)}
# input_2 range:  {np.float32(0.0), np.float32(0.6862745)}
# input_3 range:  {np.float32(0.0), np.float32(1.0)}
# input_4 range:  {np.float32(0.0), np.uint8(255)}

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage import img_as_float
from skimage.filters import threshold_niblack


def niblack_thresholding(image_rgb, window_size=25, k=-0.2):
    """
    Apply Niblack's adaptive thresholding method.

    Args:
        image_rgb (np.array): Input RGB image [0,255].
        window_size (int): Size of the local window.
        k (float): Niblack parameter.

    Returns:
        np.array: Thresholded binary image (0 or 255).
    """
    # Convert to grayscale [0,1]
    gray = rgb2gray(image_rgb)
    gray = img_as_float(gray)

    # Compute Niblack threshold
    thresh_niblack = threshold_niblack(gray, window_size=window_size, k=k)

    # Apply threshold → binary image
    binary = gray > thresh_niblack
    binary = (binary * 255).astype(np.uint8)
    
    return gray, binary


# Load inputs
try:
    input_1 = mpimg.imread('data/thresh/receipt.png')
    input_2 = mpimg.imread('data/thresh/qr.png')
    input_3 = mpimg.imread('data/thresh/blackboard.png')
    input_4 = mpimg.imread('data/thresh/lilavati.tif')
except FileNotFoundError:
    print("Check file paths.")

input_images = [input_1, input_3, input_4] #not considering input_2 === qr image
titles = ["Receipt", "Blackboard", "Lilavati"]
    
# Plotting

fig, axes = plt.subplots(len(input_images), 3, figsize=(12, 12))

# for i, img in enumerate(input_images):
#     print("shape: ", img.shape)
    
#     gray, niblack = niblack_thresholding(img, window_size=5, k=-0.2)
    
#     axes[i,0].imshow(img)
#     axes[i,0].set_title(f"Original - {titles[i]}")
#     axes[i,0].axis("off")

#     axes[i,1].imshow(gray, cmap="gray")
#     axes[i,1].set_title("Grayscale")
#     axes[i,1].axis("off")

#     axes[i,2].imshow(niblack, cmap="gray")
#     axes[i,2].set_title("Niblack Thresholded")
#     axes[i,2].axis("off")
    

for i, img in enumerate(input_images):
    print("shape: ", img.shape)
    
    gray, binary = niblack_thresholding(img, window_size=25, k=-0.2)
    
    # Compute per-pixel threshold map again here
    threshold_map = threshold_niblack(gray, window_size=25, k=-0.2)

    # Column 0: Original RGB image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Original - {titles[i]}")
    axes[i, 0].axis("off")

    # Column 1: Grayscale image
    axes[i, 1].imshow(gray, cmap="gray")
    axes[i, 1].set_title("Grayscale")
    axes[i, 1].axis("off")

    # Column 2: Per-pixel Niblack thresholds
    axes[i, 2].imshow(threshold_map, cmap="gray")
    axes[i, 2].set_title("Per-pixel Threshold Map")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
