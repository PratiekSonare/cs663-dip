import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import rotate

import numpy as np

def get_rotation_matrix(angle):
    """Return the 2D rotation matrix for a given angle (in degrees)."""
    theta = np.deg2rad(angle)
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def interpolate(image, x, y, method="nearest"):
    """Interpolate pixel value at (x, y) using nearest or bilinear."""
    h, w = image.shape[:2]

    if method == "nearest":
        x_round, y_round = int(round(x)), int(round(y))
        if 0 <= y_round < h and 0 <= x_round < w:
            return image[y_round, x_round]
        else:
            return 0  # background black

    elif method == "bilinear":
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1

        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            return 0

        # fractional parts
        dx, dy = x - x0, y - y0

        # bilinear interpolation
        I00 = image[y0, x0]
        I10 = image[y0, x1]
        I01 = image[y1, x0]
        I11 = image[y1, x1]

        return (I00 * (1 - dx) * (1 - dy) +
                I10 * dx * (1 - dy) +
                I01 * (1 - dx) * dy +
                I11 * dx * dy)

    else:
        raise ValueError("Interpolation method must be 'nearest' or 'bilinear'.")


def myImageRotation(image, angle, interpolation_method="nearest"):
    """Rotate image manually using backward mapping + interpolation."""
    h, w = image.shape[:2]
    rotated = np.zeros_like(image)

    # center of image
    cx, cy = w / 2, h / 2

    # inverse rotation matrix
    R = get_rotation_matrix(-angle)

    for y_new in range(h):
        for x_new in range(w):
            # shift to origin
            x_shift, y_shift = x_new - cx, y_new - cy

            # apply inverse rotation
            x_old, y_old = R @ np.array([x_shift, y_shift])

            # shift back
            x_src, y_src = x_old + cx, y_old + cy

            # assign interpolated value
            rotated[y_new, x_new] = interpolate(image, x_src, y_src, method=interpolation_method)

    return rotated


try:
    main_image = mpimg.imread('data/interp/main.png')
except FileNotFoundError:
    print("Check the file path.")


angle = 5

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
