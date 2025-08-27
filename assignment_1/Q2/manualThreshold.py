#1. convert all images to unit8 dtype, i.e, (0, 255) range of values
#2. average out the RGB channels for grayscale
#3. apply threshold on the grayscaled img

# some input images are float with max() <= 1 and some are already uint8;

# input_1 range:  {np.float32(0.0), np.float32(1.0)}
# input_2 range:  {np.float32(0.0), np.float32(0.6862745)}
# input_3 range:  {np.float32(0.0), np.float32(1.0)}
# input_4 range:  {np.float32(0.0), np.uint8(255)}

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def manualThresholding(image, threshold):
    """
    Performs thresholding by assigning pixels to black (0) if < threshold else white (255).

    Args:
        image_rgb (np.array): Input RGB image with values in [0, 255] or [0, 1]. If [0,1 ] conver to unit8 dtype.
        threshold (int): Threshold value.

    Returns:
        np.array: Thresholded binary image (0 or 255).
    """
    
    #Convert to unit8 dtype - (0, 255)
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:  # float in [0,1]
            image = (image * 255).astype(np.uint8)
        else:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    
    # Convert RGB -> grayscale by averaging across color channels
    if(image.ndim == 3):
        gray = np.mean(image, axis=2)
    else:
        gray = image

    # Apply threshold
    thresh = np.where(gray < threshold, 0, 255).astype(np.uint8)

    return gray, thresh


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
    gray, thresh = manualThresholding(img, 175)

    # Original
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"{titles[i]} - Original")
    axes[i, 0].axis("off")

    # Grayscale
    axes[i, 1].imshow(gray, cmap="gray")
    axes[i, 1].set_title("Grayscale")
    axes[i, 1].axis("off")

    # Thresholded
    axes[i, 2].imshow(thresh, cmap="gray")
    axes[i, 2].set_title("Thresholded")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()
