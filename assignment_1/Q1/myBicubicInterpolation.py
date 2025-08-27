import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
            
    return np.clip(new_image, 0, 1)

# Load the input image
try:
    random_image = mpimg.imread('data/interp/random.png')
except FileNotFoundError:
    print("Check the file path.")

M, N = random_image.shape
new_rows = 300 * (M - 1) + 1
new_cols = 300 * (N - 1) + 1


enlarged_bicubic = myBicubicInterpolation(random_image, new_rows, new_cols)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

im1 = axes[0].imshow(random_image, cmap='jet', aspect='equal', extent=[0, random_image.shape[1], random_image.shape[0], 0])
axes[0].set_title('Original Image')
axes[0].set_xlabel('Pixel Column')
axes[0].set_ylabel('Pixel Row')

im2 = axes[1].imshow(enlarged_bicubic, cmap='jet', aspect='equal', extent=[0, enlarged_bicubic.shape[1], enlarged_bicubic.shape[0], 0])
axes[1].set_title('Enlarged (Bicubic)')
axes[1].set_xlabel('Pixel Column')
axes[1].set_ylabel('Pixel Row')
fig.colorbar(im2, ax=axes)

plt.show()
