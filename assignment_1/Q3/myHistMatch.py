import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.color import rgb2lab, lab2rgb

def myHistMatch(source_rgb, ref_rgb, num_bins=256):
    """
    Matches the histogram of a source image to a reference image.
    Operates on L, a, and b channels independently, ignoring the black background.

    Args:
        source_rgb (np.array): The input image to be modified.
        ref_rgb (np.array): The reference image.
        num_bins (int): Number of bins for histogram calculation.

    Returns:
        np.array: The histogram-matched image.
    """
    # Convert to Lab space
    source_lab = rgb2lab(source_rgb)
    ref_lab = rgb2lab(ref_rgb)
    matched_lab = np.copy(source_lab)

    # Create masks to ignore the black background (pixels with low luminance)
    source_mask = source_lab[:, :, 0] > 5
    ref_mask = ref_lab[:, :, 0] > 5

    # Match histograms for each channel (L, a, b)
    for i in range(3):
        source_chan = source_lab[:, :, i]
        ref_chan = ref_lab[:, :, i]

        source_vals = source_chan[source_mask]
        ref_vals = ref_chan[ref_mask]

        # Calculate CDFs
        hist_source, bins = np.histogram(source_vals.ravel(), num_bins)
        hist_ref, _ = np.histogram(ref_vals.ravel(), bins=num_bins, range=(bins[0], bins[-1]))
        
        cdf_source = hist_source.cumsum()
        cdf_ref = hist_ref.cumsum()

        # Normalize CDFs
        cdf_source = cdf_source / cdf_source.max()
        cdf_ref = cdf_ref / cdf_ref.max()

        # Create the mapping function
        mapping_func = np.interp(cdf_source, cdf_ref, np.arange(num_bins))

        # Apply the mapping to the source channel (only the foreground)
        matched_chan = np.interp(source_chan[source_mask], np.arange(num_bins), mapping_func)
        
        # Put the matched values back into the image
        temp_chan = matched_lab[:, :, i]
        temp_chan[source_mask] = matched_chan
        matched_lab[:, :, i] = temp_chan

    # Convert back to RGB
    matched_rgb = lab2rgb(matched_lab)
    return np.clip(matched_rgb, 0, 1)

# Load images
try:
    retina_image = mpimg.imread('data/hist/retina.png')
    retina_ref = mpimg.imread('data/hist/retinaRef.png')
except FileNotFoundError:
    print("Error: retina images not found. Please check file paths.")
    retina_image = np.random.rand(256, 256, 3)
    retina_ref = np.random.rand(256, 256, 3)
    
# Perform histogram matching
matched_image = myHistMatch(retina_image, retina_ref)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Images
axes[0, 0].imshow(retina_image)
axes[0, 0].set_title('Original Image')
axes[0, 1].imshow(retina_ref)
axes[0, 1].set_title('Reference Image')
axes[0, 2].imshow(matched_image)
axes[0, 2].set_title('Matched Image')

# Histograms (Luminance channel)
axes[1, 0].hist(rgb2lab(retina_image)[:,:,0][rgb2lab(retina_image)[:,:,0] > 5].ravel(), bins=256, color='red', alpha=0.7)
axes[1, 0].set_title('Original L* Histogram')

axes[1, 1].hist(rgb2lab(retina_ref)[:,:,0][rgb2lab(retina_ref)[:,:,0] > 5].ravel(), bins=256, color='blue', alpha=0.7)
axes[1, 1].set_title('Reference L* Histogram')

axes[1, 2].hist(rgb2lab(matched_image)[:,:,0][rgb2lab(matched_image)[:,:,0] > 5].ravel(), bins=256, color='green', alpha=0.7)
axes[1, 2].set_title('Matched L* Histogram')

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')
for ax in axes[1]:
    ax.axis('on')

plt.tight_layout()
plt.show()