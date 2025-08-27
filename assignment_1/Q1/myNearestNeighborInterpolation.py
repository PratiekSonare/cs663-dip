import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def myNearestNeighborInterpolation(image, new_rows, new_cols):
    """
    Enlarges an image using nearest-neighbor interpolation.

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
            old_i = int(round(i * row_ratio))
            old_j = int(round(j * col_ratio))
            new_image[i, j] = image[old_i, old_j]
            
    return new_image

# Load the input image
try:
    random_image = mpimg.imread('data/interp/random.png')
except FileNotFoundError:
    print("check the file path.")

# new row and col sizes
M, N = random_image.shape
new_rows = 300 * (M - 1) + 1
new_cols = 300 * (N - 1) + 1

enlarged_nn = myNearestNeighborInterpolation(random_image, new_rows, new_cols)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# og image
im1 = axes[0].imshow(random_image, cmap='jet', aspect='equal', extent=[0, random_image.shape[1], random_image.shape[0], 0])
axes[0].set_title('Original Image')
axes[0].set_xlabel('Pixel Column')
axes[0].set_ylabel('Pixel Row')
fig.colorbar(im1, ax=axes[0])

# Enlarged Image
im2 = axes[1].imshow(enlarged_nn, cmap='jet', aspect='equal', extent=[0, enlarged_nn.shape[1], enlarged_nn.shape[0], 0])
axes[1].set_title('Enlarged (Nearest Neighbor)')
axes[1].set_xlabel('Pixel Column')
axes[1].set_ylabel('Pixel Row')
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
