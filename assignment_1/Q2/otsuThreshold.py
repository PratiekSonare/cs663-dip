#1. convert all images to unit8 dtype, i.e, (0, 255) range of values
#2. average out the RGB channels for grayscale
#3. apply otsu threshold from scikit-image on the grayscaled image

# some input images are float with max() <= 1 and some are already uint8;

# input_1 range:  {np.float32(0.0), np.float32(1.0)}
# input_2 range:  {np.float32(0.0), np.float32(0.6862745)}
# input_3 range:  {np.float32(0.0), np.float32(1.0)}
# input_4 range:  {np.float32(0.0), np.uint8(255)}


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.filters import threshold_otsu

def otsuThresholding(image):
    """
    Applies Otsu's thresholding on an input image.
    Steps:
    1. Convert to uint8 (0–255 range).
    2. Convert to grayscale (average RGB channels).
    3. Compute Otsu threshold and binarize.
    
    Args:
        image (np.array): Input RGB or grayscale image (float [0,1] or uint8).
    
    Returns:
        gray (np.array): Grayscale image (uint8).
        binary (np.array): Otsu thresholded binary image (uint8).
        thresh_value (int): Chosen Otsu threshold.
    """
    # Step 1: Ensure uint8
    # Step 1: Ensure uint8
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:  # float in [0,1]
            image = (image * 255).astype(np.uint8)
        else:  # arbitrary float range
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    
    # Step 2: Convert to grayscale
    if image.ndim == 3:  # RGB or RGBA
        gray = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = image
    
    # Step 3: Otsu threshold
    thresh_value = threshold_otsu(gray)
    binary = np.where(gray < thresh_value, 0, 255).astype(np.uint8)
    
    return gray, binary, thresh_value


# ---- Load images ----
try:
    input_1 = mpimg.imread('data/thresh/receipt.png')
    input_2 = mpimg.imread('data/thresh/qr.png')
    input_3 = mpimg.imread('data/thresh/blackboard.png')
    input_4 = mpimg.imread('data/thresh/lilavati.tif')
except FileNotFoundError:
    print("Check file paths.")

input_images = [input_1, input_2, input_3, input_4]
titles = ["Receipt", "QR Code", "Blackboard", "Lilavati"]

# ---- Visualization ----
fig, axes = plt.subplots(4, 3, figsize=(12, 12))

for i, img in enumerate(input_images):
    gray, binary, t_val = otsuThresholding(img)

    # Original
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"{titles[i]} - Original")
    axes[i, 0].axis("off")

    # Grayscale
    axes[i, 1].imshow(gray, cmap="gray")
    axes[i, 1].set_title("Grayscale")
    axes[i, 1].axis("off")

    # Thresholded
    axes[i, 2].imshow(binary, cmap="gray")
    axes[i, 2].set_title(f"Otsu Thresholded (T={t_val})")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
