import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def myImageShrink(image, d):
    return image[::d, ::d]

def myNearestNeighborInterpolation(image, new_rows, new_cols):
    old_rows, old_cols = image.shape
    row_ratio = (old_rows - 1) / (new_rows - 1)
    col_ratio = (old_cols - 1) / (new_cols - 1)
    
    new_image = np.zeros((new_rows, new_cols))
    
    for i in range(new_rows):
        for j in range(new_cols):
            old_i = int(round(i * row_ratio))
            old_j = int(round(j * col_ratio))
            new_image[i, j] = image[old_i, old_j]
            
    return new_image

def myBilinearInterpolation(image, new_rows, new_cols):
    """
    Enlarges an image using bilinear interpolation.

    Args:
        image (np.array): The input image.
        new_rows (int): The number of rows in the new image.
        new_cols (int): The number of columns in the new image.

    Returns:
        np.array: The enlarged image.
    """
    old_rows, old_cols = image.shape
    row_ratio = (old_rows - 1) / (new_rows - 1)
    col_ratio = (old_cols - 1) / (new_cols - 1)
    
    new_image = np.zeros((new_rows, new_cols))
    
    for i in range(new_rows):
        for j in range(new_cols):
            x = i * row_ratio
            y = j * col_ratio
            x1, y1 = int(np.floor(x)), int(np.floor(y))
            x2, y2 = min(x1 + 1, old_rows - 1), min(y1 + 1, old_cols - 1)
            
            dx = x - x1
            dy = y - y1
            
            f_x1y1 = image[x1, y1]
            f_x2y1 = image[x2, y1]
            f_x1y2 = image[x1, y2]
            f_x2y2 = image[x2, y2]
            
            f_xy = (f_x1y1 * (1 - dx) * (1 - dy) +
                    f_x2y1 * dx * (1 - dy) +
                    f_x1y2 * (1 - dx) * dy +
                    f_x2y2 * dx * dy)
            
            new_image[i, j] = f_xy
            
    return new_image

def myBicubicInterpolation(image, new_rows, new_cols):
    old_rows, old_cols = image.shape
    row_ratio = (old_rows - 1) / (new_rows - 1)
    col_ratio = (old_cols - 1) / (new_cols - 1)
    
    new_image = np.zeros((new_rows, new_cols))
    
    def cubic_kernel(x):
        x = np.abs(x)
        if x <= 1:
            return 1 - 2 * x**2 + x**3
        elif x <= 2:
            return 4 - 8 * x + 5 * x**2 - x**3
        else:
            return 0

    for i in range(new_rows):
        for j in range(new_cols):
            x = i * row_ratio
            y = j * col_ratio
            x_int, y_int = int(np.floor(x)), int(np.floor(y))
            dx = x - x_int
            dy = y - y_int
            
            value = 0
            for m in range(-1, 3):
                for n in range(-1, 3):
                    # ADDED: Calculate and clip coordinates to stay within original image bounds
                    ix = np.clip(x_int + m, 0, old_rows - 1)
                    iy = np.clip(y_int + n, 0, old_cols - 1)
                    
                    # MODIFIED: Access the original image with safe coordinates
                    px = image[ix, iy]

                    wx = cubic_kernel(m - dx)
                    wy = cubic_kernel(n - dy)
                    value += px * wx * wy
            new_image[i, j] = value
            
    return new_image

# Load the input image
try:
    suit_image = mpimg.imread('data/interp/random.png')
except FileNotFoundError:
    raise FileNotFoundError("Error: check file path!")

M, N = suit_image.shape
new_rows = 300 * (M - 1) + 1
new_cols = 300 * (N - 1) + 1

nn_image = myNearestNeighborInterpolation(suit_image, new_rows, new_cols)
bilinear_image = myBilinearInterpolation(suit_image, new_rows, new_cols)
bicubic_image = myBicubicInterpolation(suit_image, new_rows, new_cols)

fig, axes = plt.subplots(1, 4, figsize=(15, 5))

im1 = axes[0].imshow(suit_image, cmap='jet')
axes[0].set_title("Original Image")
# axes[0].axis("off")

im2 = axes[1].imshow(nn_image, cmap='jet')
axes[1].set_title("Nearest Neighbor")
# axes[1].axis("off")

im3 = axes[2].imshow(bilinear_image, cmap='jet')
axes[2].set_title("Bilinear Interpolation")
# axes[2].axis("off")

im4 = axes[3].imshow(bicubic_image, cmap='jet')
axes[3].set_title("Bicubic Interpolation")
# axes[3].axis("off")

# Add a single colorbar for all
fig.colorbar(im4, ax=axes, orientation="vertical", fraction=0.02)

plt.show()
