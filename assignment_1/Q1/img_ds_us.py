import numpy as np
import matplotlib.pyplot as plt
import scipy.io

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

def calculate_rmse(actual, interpolated):
    return np.sqrt(np.mean((actual.astype(np.float64) - interpolated.astype(np.float64))**2))

try:
    mat_data = scipy.io.loadmat('data/interp/ct.mat')
    original_ct = mat_data['original']
    subsampled_ct = mat_data['subsampled']

    new_rows, new_cols = original_ct.shape

    # Enlarging with interpolation
    enlarged_nn_ct = myNearestNeighborInterpolation(subsampled_ct, new_rows, new_cols)
    enlarged_bilinear_ct = myBilinearInterpolation(subsampled_ct, new_rows, new_cols)
    enlarged_bicubic_ct = myBicubicInterpolation(subsampled_ct, new_rows, new_cols)

    # difference
    diff_nn = original_ct - enlarged_nn_ct
    diff_bilinear = original_ct - enlarged_bilinear_ct
    diff_bicubic = original_ct - enlarged_bicubic_ct

    rmse_nn = calculate_rmse(original_ct, enlarged_nn_ct)
    rmse_bilinear = calculate_rmse(original_ct, enlarged_bilinear_ct)
    rmse_bicubic = calculate_rmse(original_ct, enlarged_bicubic_ct)
    print(f"--- Final RMSE Values ---")
    print(f"RMSE (Nearest Neighbor): {rmse_nn:.4f}")
    print(f"RMSE (Bilinear): {rmse_bilinear:.4f}")
    print(f"RMSE (Bicubic): {rmse_bicubic:.4f}")


    fig, axes = plt.subplots(1, 4, figsize=(20, 10))

    axes[0].set_title('Original')
    im = axes[0].imshow(original_ct, cmap='jet')

    # axes[1].set_title(f'Nearest Neighbor\nRMSE: {rmse_nn:.2f}')
    # im = axes[1].imshow(enlarged_nn_ct, cmap='jet')

    # axes[2].set_title(f'Bilinear\nRMSE: {rmse_bilinear:.2f}')
    # im = axes[2].imshow(enlarged_bilinear_ct, cmap='jet')

    # axes[3].set_title(f'Bicubic\nRMSE: {rmse_bicubic:.2f}')
    # im = axes[3].imshow(enlarged_bicubic_ct, cmap='jet')
    # fig.colorbar(im, ax=axes, orientation='vertical')

    # plt.show()

    # axes[1, 0].axis('off')

    axes[1].set_title('Difference (NN)')
    im = axes[1].imshow(diff_nn, cmap='jet')

    axes[2].set_title('Difference (Bilinear)')
    im = axes[2].imshow(diff_bilinear, cmap='jet')

    axes[3].set_title('Difference (Bicubic)')
    im = axes[3].imshow(diff_bicubic, cmap='jet')
    
    fig.colorbar(im, ax=axes, orientation='vertical')

    plt.show()

except FileNotFoundError:
    print("Check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
