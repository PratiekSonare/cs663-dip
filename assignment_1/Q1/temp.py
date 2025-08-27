import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    suit_image = mpimg.imread('data/interp/suit.png')
except FileNotFoundError:
    print("Error: check file path!")

print(suit_image)