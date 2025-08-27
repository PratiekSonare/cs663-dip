import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage import exposure, color, img_as_ubyte

# Load the input image
try:
    canyon_image = mpimg.imread('data/hist/canyon.png')
    if canyon_image.dtype != np.uint8:
        canyon_image = (canyon_image * 255).astype(np.uint8)
except FileNotFoundError:
    print("Check the file path.")

#skimage converts l_channel from [0-255] to [0-100], and to apply clahe, we must normalize it to [0-1]
#hence we return l_channel in the range [0-100] to retain the luminance and chroma channels

canyon_lab = color.rgb2lab(canyon_image)
l_channel = canyon_lab[:, :, 0]   # [0,100]
a_channel = canyon_lab[:, :, 1]   # ~[-128,127]
b_channel = canyon_lab[:, :, 2]   # ~[-128,127]

def apply_clahe_skimage(l_in, a_in, b_in, clip_limit, grid_size):
    """
    Applies CLAHE using skimage to the L channel.
    Returns processed RGB image and new L channel.
    
    - clip_limit: float (0–1, lower = less contrast, higher = more contrast)
    - grid_size: int or (int,int), size of contextual regions
    """
    # Normalize L to [0,1] for skimage.adapthist
    l_norm = l_in / 100.0  

    # Apply CLAHE
    l_eq = exposure.equalize_adapthist(l_norm, clip_limit=clip_limit, kernel_size=grid_size)

    # Rescale back to [0,100] for Lab
    l_out = l_eq * 100  

    # Recombine Lab channels
    lab_eq = np.zeros_like(canyon_lab)
    lab_eq[:, :, 0] = l_out
    lab_eq[:, :, 1] = a_in
    lab_eq[:, :, 2] = b_in

    # Convert Lab -> RGB
    rgb_eq = color.lab2rgb(lab_eq)
    rgb_eq_ubyte = img_as_ubyte(rgb_eq)
    return rgb_eq_ubyte, l_out

# Generate results with skimage
tuned_result, tuned_l = apply_clahe_skimage(l_channel, a_channel, b_channel, clip_limit=0.02, grid_size=(8, 8))
large_grid_result, large_l = apply_clahe_skimage(l_channel, a_channel, b_channel, clip_limit=0.02, grid_size=(64, 64))
small_grid_result, small_l = apply_clahe_skimage(l_channel, a_channel, b_channel, clip_limit=0.02, grid_size=(2, 2))
low_clip_result, low_clip_l = apply_clahe_skimage(l_channel, a_channel, b_channel, clip_limit=0.005, grid_size=(8, 8))

# --- FIGURE 1: Images ---
fig1, axes1 = plt.subplots(1, 5, figsize=(25, 8))
images = [canyon_image, tuned_result, large_grid_result, small_grid_result, low_clip_result]
titles = ['Original', 
          'Tuned (Clip=0.02, Grid=8x8)', 
          'Large Grid (Clip=0.02, Grid=64x64)', 
          'Small Grid (Clip=0.02, Grid=2x2)', 
          'Low Clip (Clip=0.005, Grid=8x8)']

for i, (img, title) in enumerate(zip(images, titles)):
    axes1[i].imshow(img)
    axes1[i].set_title(title)
    axes1[i].axis('off')

plt.tight_layout()
plt.show()

# --- FIGURE 2: Histograms ---
fig2, axes2 = plt.subplots(1, 5, figsize=(25, 6))
l_channels = [l_channel, tuned_l, large_l, small_l, low_clip_l]

for i, l in enumerate(l_channels):
    axes2[i].hist(l.ravel(), bins=256, range=[0, 101], color='black')
    axes2[i].set_title(f"{titles[i]} Histogram")
    axes2[i].set_xlim([0, 101])

plt.tight_layout()
plt.show()
