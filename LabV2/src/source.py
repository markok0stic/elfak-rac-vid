import cv2 as cv
import numpy as np

# Load the input image
image = cv.imread('coins.png')

# Convert the image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply a threshold to the grayscale image to convert it into a binary image
_, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)

# Define a kernel for morphological operations
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))

# Perform morphological dilation, erosion, closing, and opening operations on the binary image
closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

# Find contours (boundaries of connected regions) in the opened binary image
cnts = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Create a mask from the filtered binary image to keep only the regions of interest
mask = np.zeros_like(closing)

# Iterate over the detected contours and filter out objects with the same color
for c in cnts:
    # Compute the average color of the object
    mask.fill(0)
    cv.drawContours(mask, [c], -1, 255, -1)
    avg_color = cv.mean(image, mask=mask)[:3]

    # Check if the color of the object is significantly different from the background color
    bg_color = [128, 128, 128]
    color_diff = np.linalg.norm(np.array(avg_color) - np.array(bg_color))
    if color_diff > 50:
        cv.drawContours(mask, [c], -1, 255, -1)

# Show the mask
cv.imshow('coin_mask.png', mask)
cv.imwrite('coin_mask.png', mask)

# Apply the mask to the original image to obtain the masked image
result = cv.bitwise_and(image, image, mask=mask)

# Show the masked image
cv.imshow('masked_coin', result)

cv.waitKey(0)
cv.destroyAllWindows()