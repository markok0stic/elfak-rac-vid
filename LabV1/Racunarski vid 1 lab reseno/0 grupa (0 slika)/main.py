import cv2
import numpy as np
import matplotlib.pyplot as plt


def crniKvadrati(matrix, x, y):
    for i in range(-2, 2):
        for j in range(-2, 2):
            matrix[x + i, y + j] = 0


# Load gray image
img = cv2.imread('input.png', 0)

# Perform 2D Fourier Transform
f = np.fft.fft2(img)

# Shift the zero-frequency component to the center of the spectrum
fshift = np.fft.fftshift(f)

# Compute the magnitude spectrum (amplitude)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra pre uklanjanja šuma')
plt.savefig('fft_mag.png')
plt.show()


# ovo rucno gledam samo gde su beli pixeli (4 ih ima)
# ovo su koordinate tih pixela
y1, x1 = 236, 156
y2, x2 = 276, 156
y3, x3 = 236, 356
y4, x4 = 276, 356

crniKvadrati(fshift, x1, y1)
crniKvadrati(fshift, x2, y2)
crniKvadrati(fshift, x3, y3)
crniKvadrati(fshift, x4, y4)

# Compute the magnitude spectrum (amplitude)
magnitude_spectrum = 20 * np.log(np.abs(fshift))

# Display the magnitude spectrum
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnituda spektra nakon uklanjanja šuma')
plt.savefig('fft_mag_filtered.png')
plt.show()


# Shift the zero-frequency component back to the top-left corner
f_ishift = np.fft.ifftshift(fshift)

# Compute the inverse Fourier Transform
img_filtered = np.fft.ifft2(f_ishift).real

# Display the filtered image

cv2.imshow('Final image', img_filtered.astype(np.uint8))
cv2.imwrite('output.png', img_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()
