import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.filters import threshold_niblack


def niblackThresholding(image, window_size=15, k=-0.2):
    """
    Apply Niblack's local thresholding.

    Args:
        image (np.array): Input RGB/grayscale image with values in [0,255] or [0,1].
        window_size (int): Size of the local neighborhood window.
        k (float): Niblack parameter (default -0.2).

    Returns:
        gray (np.array): Grayscale image.
        binary (np.array): Thresholded image (0 or 255).
    """
    # Convert to uint8 (0–255)
    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

    # Convert to grayscale
    if image.ndim == 3:
        gray = np.mean(image[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = image

    # Apply Niblack threshold
    thresh = threshold_niblack(gray, window_size=window_size, k=k)
    binary = np.where(gray > thresh, 255, 0).astype(np.uint8)

    return gray, binary


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

window_sizes = [5, 15, 25]

# ---- Visualization ----
fig, axes = plt.subplots(len(input_images), len(window_sizes), figsize=(12, 12))

for i, img in enumerate(input_images):
    for j, ws in enumerate(window_sizes):
        gray, binary = niblackThresholding(img, window_size=ws, k=-0.2)
        axes[i, j].imshow(binary, cmap="gray")
        axes[i, j].set_title(f"{titles[i]}\nWin={ws}")
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()
