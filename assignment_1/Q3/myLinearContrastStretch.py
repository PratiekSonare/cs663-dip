import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2hsv, hsv2rgb

def myLinearContrastStretch(image_rgb):
    """
    Performs linear contrast stretching on the luminance (V) component of an RGB image.

    Args:
        image_rgb (np.array): Input RGB image with values in [0, 1].

    Returns:
        np.array: Contrast-enhanced RGB image.
    """
    # Convert to HSV color space
    image_hsv = rgb2hsv(image_rgb)
    v_channel = image_hsv[:, :, 2]

    # Find the min and max intensity values
    v_min = v_channel.min()
    v_max = v_channel.max()

    # Apply the linear stretch formula
    stretched_v = (v_channel - v_min) / (v_max - v_min)

    # Clip values to be within the valid range [0, 1] after stretching
    stretched_v = np.clip(stretched_v, 0, 1)
    
    print("Original V range:", v_channel.min(), v_channel.max())
    print("Stretched V range:", stretched_v.min(), stretched_v.max())
    
    # Create the new HSV image and convert back to RGB
    stretched_hsv = image_hsv.copy()
    stretched_hsv[:, :, 2] = stretched_v
    stretched_rgb = hsv2rgb(stretched_hsv)
    
    return stretched_rgb

def myLinearContrastStretchMask(image_rgb):
    """
    Performs linear contrast stretching on the luminance (V) component of an RGB image.

    Args:
        image_rgb (np.array): Input RGB image with values in [0, 1].

    Returns:
        np.array: Contrast-enhanced RGB image.
    """
    # Convert to HSV color space
    image_hsv = rgb2hsv(image_rgb)
    v_channel = image_hsv[:, :, 2]

    # Create a mask for pixels with V <= 0.5
    mask = v_channel <= 0.5

    # Find the min and max intensity values using only the masked pixels
    v_min = v_channel[mask].min()
    v_max = v_channel[mask].max()

    # Apply the linear stretch formula
    stretched_v = (v_channel - v_min) / (v_max - v_min)

    # Clip values to be within the valid range [0, 1] after stretching
    stretched_v = np.clip(stretched_v, 0, 1)

    print("Original V range (after masking):", v_min, v_max)
    print("Stretched V range:", stretched_v.min(), stretched_v.max())
    
    # Create the new HSV image and convert back to RGB
    stretched_hsv = image_hsv.copy()
    stretched_hsv[:, :, 2] = stretched_v
    stretched_rgb = hsv2rgb(stretched_hsv)
    
    return stretched_rgb

# Load the input image
try:
    leh_image = mpimg.imread('data/hist/leh.png')
        
except FileNotFoundError:
    print("Check the file path.")

# Perform contrast stretching
stretched_image = myLinearContrastStretch(leh_image)
stretched_image_mask = myLinearContrastStretchMask(leh_image)

# Display the results
fig, axes = plt.subplots(2, 3, figsize=(12, 10))

# Original Image and its Histogram
axes[0, 0].imshow(leh_image)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[1, 0].hist(rgb2hsv(leh_image)[:, :, 2].ravel(), bins=256, color='gray')
axes[1, 0].set_xlim([0, 1])
axes[1, 0].set_title('Original Luminance Histogram')
axes[1, 0].set_xlabel('Intensity')
axes[1, 0].set_ylabel('Pixel Count')

# Enhanced Image and its Histogram
axes[0, 1].imshow(stretched_image)
axes[0, 1].set_title('contrast-stretched image')
axes[0, 1].axis('off')

# Enhanced Image and its Histogram
axes[0, 2].imshow(stretched_image_mask)
axes[0, 2].set_title('contrast stretched image (after masking)')
axes[0, 2].axis('off')

axes[1, 1].hist(rgb2hsv(stretched_image)[:, :, 2].ravel(), bins=256, color='gray')
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_title('Stretched Luminance Histogram')
axes[1, 1].set_xlabel('Intensity')
axes[1, 1].set_ylabel('Pixel Count')

axes[1, 2].hist(rgb2hsv(stretched_image_mask)[:, :, 2].ravel(), bins=256, color='gray')
axes[1, 2].set_xlim([0, 1])
axes[1, 2].set_title('Stretched Luminance Histogram (after masking)')
axes[1, 2].set_xlabel('Intensity')
axes[1, 2].set_ylabel('Pixel Count')

plt.tight_layout()
plt.show()