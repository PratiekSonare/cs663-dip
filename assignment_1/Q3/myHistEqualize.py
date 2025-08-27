import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2hsv, hsv2rgb

import numpy as np
from skimage.color import rgb2hsv, hsv2rgb

def myHistEqualize(image_rgb, num_bins=256, mask_threshold=0.5):
    """
    Performs histogram equalization on the luminance (V) component of an RGB image,
    but only on pixels below a threshold (mask).

    Args:
        image_rgb (np.array): Input RGB image with values in [0, 1].
        num_bins (int): Number of bins for histogram.
        mask_threshold (float): Pixels with V <= threshold are equalized.

    Returns:
        np.array: RGB image with masked histogram equalization applied.
    """
    # Convert to HSV
    image_hsv = rgb2hsv(image_rgb)
    v_channel = image_hsv[:, :, 2]

    # Define mask
    mask = v_channel <= mask_threshold
    v_masked = v_channel[mask]  # 1D values to process

    # Histogram Equalization on masked pixels only
    if v_masked.size > 0:
        v_int = np.round(v_masked * (num_bins - 1)).astype(int)
        hist, _ = np.histogram(v_int, bins=num_bins, range=(0, num_bins - 1))
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()

        # Map original intensities through equalization
        equalized_v_int = np.interp(v_int, np.arange(num_bins), cdf * (num_bins - 1) / cdf.max())
        equalized_v = equalized_v_int / (num_bins - 1)

        # Put back into the V channel
        v_channel_new = v_channel.copy()
        v_channel_new[mask] = equalized_v
    else:
        v_channel_new = v_channel.copy()  # nothing to change

    # Reconstruct HSV → RGB
    equalized_hsv = image_hsv.copy()
    equalized_hsv[:, :, 2] = v_channel_new
    equalized_rgb = hsv2rgb(equalized_hsv)

    return equalized_rgb

# Load the input image
try:
    leh_image = mpimg.imread('data/hist/leh.png')
except FileNotFoundError:
    print("Check the file path.")

# Perform histogram equalization
he_image = myHistEqualize(leh_image)

# Display the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original Image and its Histogram
axes[0, 0].imshow(leh_image)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[1, 0].hist(rgb2hsv(leh_image)[:, :, 2].ravel(), bins=256, color='gray')
axes[1, 0].set_title('Original Luminance Histogram')

# Enhanced Image and its Histogram
axes[0, 1].imshow(he_image)
axes[0, 1].set_title('Histogram-Equalized Image')
axes[0, 1].axis('off')

axes[1, 1].hist(rgb2hsv(he_image)[:, :, 2].ravel(), bins=256, color='gray')
axes[1, 1].set_title('Equalized Luminance Histogram')

plt.tight_layout()
plt.show()