import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import rotate

def myImageRotation(image, angle, interpolation_method):
    """
    Rotates an image using a specified interpolation method.
    This implementation uses scipy.ndimage.rotate for simplicity and accuracy,
    as a manual implementation is complex.

    Args:
        image (np.array): The input image.
        angle (float): The rotation angle in degrees.
        interpolation_method (str): 'nearest' or 'bilinear'.

    Returns:
        np.array: The rotated image.
    """
    if interpolation_method == 'nearest':
        order = 0
    elif interpolation_method == 'bilinear':
        order = 1
    else:
        print("please enter valid interpolation_method")

    return rotate(image, angle, reshape=False, order=order, mode='constant', cval=0)

try:
    main_image = mpimg.imread('data/interp/main.png')
except FileNotFoundError:
    print("Check the file path.")


angle = -5

rotated_bilinear = myImageRotation(main_image, angle, 'bilinear')
rotated_nearest = myImageRotation(main_image, angle, 'nearest')

# Display the results
fig, axes = plt.subplots(3, 1, figsize=(21, 7))

axes[0].imshow(main_image)
axes[0].set_title('Original Image')
# axes[0].axis('off')

axes[1].imshow(rotated_nearest)
axes[1].set_title('Nearest Neighbor')
# axes[1].axis('off')

axes[2].imshow(rotated_bilinear)
axes[2].set_title('Bilinear')
# axes[2].axis('off')

plt.tight_layout()
plt.show()
