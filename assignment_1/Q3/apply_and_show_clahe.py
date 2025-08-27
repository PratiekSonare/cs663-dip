import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 # Using OpenCV for a standard CLAHE implementation

# Load the input image
try:
    canyon_image = mpimg.imread('data/hist/canyon.png')
    # Ensure image is in uint8 format for OpenCV
    if canyon_image.dtype != np.uint8:
        canyon_image = (canyon_image * 255).astype(np.uint8)
except FileNotFoundError:
    print("Check the file path.")

# Convert to Lab color space for better luminance processing
canyon_lab = cv2.cvtColor(canyon_image, cv2.COLOR_RGB2Lab)
l_channel, a_channel, b_channel = cv2.split(canyon_lab)

def apply_clahe(l_in, a_in, b_in, clip_limit, grid_size):
    """
    Applies the CLAHE algorithm to the L channel and returns the resulting RGB image.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_out = clahe.apply(l_in)
    
    clahe_lab = cv2.merge((l_out, a_in, b_in))
    clahe_rgb = cv2.cvtColor(clahe_lab, cv2.COLOR_Lab2RGB)
    return clahe_rgb

# --- 1. Generate all the CLAHE results first ---

# Tuned Parameters
tuned_result = apply_clahe(l_channel, a_channel, b_channel, 
                           clip_limit=2.0, grid_size=(8, 8))

# Large Neighborhood
large_grid_result = apply_clahe(l_channel, a_channel, b_channel, 
                                clip_limit=2.0, grid_size=(64, 64))

# Small Neighborhood
small_grid_result = apply_clahe(l_channel, a_channel, b_channel, 
                                clip_limit=2.0, grid_size=(2, 2))

# Lower Clip Limit
low_clip_result = apply_clahe(l_channel, a_channel, b_channel, 
                              clip_limit=1.0, grid_size=(8, 8))

# --- 2. Create a single figure and plot all images ---

# Create a 2x4 grid of subplots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# --- Row 1 ---

# Original Image
axes[0, 0].imshow(canyon_image)
axes[0, 0].set_title('Original Image')

# Tuned Result
axes[0, 1].imshow(tuned_result)
axes[0, 1].set_title('Tuned (Clip=2.0, Grid=8x8)')

# Large Grid Result
axes[0, 2].imshow(large_grid_result)
axes[0, 2].set_title('Large Grid (64x64) - Low Contrast')

# Small Grid Result
axes[0, 3].imshow(small_grid_result)
axes[0, 3].set_title('Small Grid (2x2) - Noisy')

# --- Row 2 ---

# The assignment asks for three additional comparisons, so we'll fill the second row
# We can show the original again for context if needed, or leave some blank.
# Here, we'll display the final result and leave others blank.

axes[1, 0].imshow(canyon_image)
axes[1, 0].set_title('Original Image')

axes[1, 1].imshow(low_clip_result)
axes[1, 1].set_title('Half Clip Limit (Clip=1.0, Grid=8x8)')

# Hide unused subplots
axes[1, 2].axis('off')
axes[1, 3].axis('off')


# --- Final Touches ---

# Turn off the axis ticks for all images
for ax in axes.ravel():
    ax.axis('off')

# Adjust layout to prevent titles from overlapping
plt.tight_layout()

# Show the single plot window
plt.show()