import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def anullPixels(matrix, x, y):
    for i in range(-2, 2):
        for j in range(-2, 2):
            matrix[x + i, y + j] = 0


# Load gray image
img = cv.imread('../assets/input.png', 0)

# Perform 2D Fourier Transform
f = np.fft.fft2(img)

# Shift the zero-frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum (amplitude)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude spectrum before filtering')
plt.savefig('fft_mag.png')
plt.show()


# manual determination of pixel coords
# possible solution for automation would be thresholding
x1, y1 = 206, 206
x2, y2 = 306, 206
x3, y3 = 206, 306
x4, y4 = 306, 306

anullPixels(fshift, x1, y1)
anullPixels(fshift, x2, y2)
anullPixels(fshift, x3, y3)
anullPixels(fshift, x4, y4)

# Compute the magnitude spectrum (amplitude)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude spectrum after filtering')
plt.savefig('fft_mag_filtered.png')
plt.show()

# Shift the zero-frequency component back to the top-left corner
f_ishift = np.fft.ifftshift(fshift)

# Compute the inverse Fourier Transform
img_filtered = np.fft.ifft2(f_ishift).real

# Display the filtered image
cv.imshow('Final image', img_filtered.astype(np.uint8))
cv.imwrite('output.png', img_filtered)
cv.waitKey(0)
cv.destroyAllWindows()
