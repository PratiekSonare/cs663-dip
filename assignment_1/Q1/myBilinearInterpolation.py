import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# Load the input image
try:
    random_image = mpimg.imread('data/interp/random.png')
except FileNotFoundError:
    print("Check the file path.")

# Define new dimensions
M, N = random_image.shape
new_rows = 300 * (M - 1) + 1
new_cols = 300 * (N - 1) + 1

# Enlarge the image
enlarged_bilinear = myBilinearInterpolation(random_image, new_rows, new_cols)

# Display the results
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

im1 = axes[0].imshow(
    random_image,
    cmap='jet',
    aspect='equal',
    extent=[0, random_image.shape[1], random_image.shape[0], 0]
)
axes[0].set_title('Original Image')
axes[0].set_xlabel('Pixel Column')
axes[0].set_ylabel('Pixel Row')

im2 = axes[1].imshow(
    enlarged_bilinear,
    cmap='jet',
    aspect='equal',
    extent=[0, enlarged_bilinear.shape[1], enlarged_bilinear.shape[0], 0]
)
axes[1].set_title('Enlarged (Bilinear)')
axes[1].set_xlabel('Pixel Column')
axes[1].set_ylabel('Pixel Row')

fig.colorbar(im2, ax=axes)
plt.show()
