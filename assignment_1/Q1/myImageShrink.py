import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def myImageShrink(image, d):
    """
    Shrinks an image by a factor of d using subsampling.

    Args:
        image (np.array): The input image.
        d (int): The downsampling factor.

    Returns:
        np.array: The shrunken image.
    """
    return image[::d, ::d]

# Load the input image
try:
    suit_image = mpimg.imread('data/interp/suit.png')
except FileNotFoundError:
    print("Error: check file path!")

# shrinking image by factors of 2 and 3 by sampling every 2th and 3rd pixel only
image_d2 = myImageShrink(suit_image, 2)
image_d3 = myImageShrink(suit_image, 3)

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# og image
im1 = axes[0].imshow(suit_image, aspect='equal')
axes[0].set_title('Original Image')
axes[0].set_xlabel('Pixel Column')
axes[0].set_ylabel('Pixel Row')

# Shrunken Image (d=2)
im2 = axes[1].imshow(image_d2, aspect='equal')
axes[1].set_title('Shrunken Image (d=2)')
axes[1].set_xlabel('Pixel Column')
axes[1].set_ylabel('Pixel Row')

# Shrunken Image (d=3)
im3 = axes[2].imshow(image_d3, aspect='equal')
axes[2].set_title('Shrunken Image (d=3)')
axes[2].set_xlabel('Pixel Column')
axes[2].set_ylabel('Pixel Row')
fig.colorbar(im3, ax=axes, orientation='vertical')

plt.show()
