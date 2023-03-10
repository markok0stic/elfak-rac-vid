import cv2
import numpy as np

# Učitavanje slike
img = cv2.imread('coins.png')

# Konvertovanje slike u grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prag segmentacija za izdvajanje novčića
th, threshed = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Morfološke operacije za popunjavanje rupa
kernel = np.ones((5,5),np.uint8)
closed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

# Filtriranje viškova
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

# Konvertovanje slike u HSV prostor boja
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Segmentacija markera na Saturation kanalu
marker = cv2.inRange(hsv, (0, 0, 100), (255, 255, 150))

# Filtriranje nepotrebnih piksela
filtered = cv2.bitwise_and(opened, marker)

# Morfološka rekonstrukcija za izdvajanje bakarnog novčića
kernel = np.ones((3,3),np.uint8)
marker = cv2.erode(filtered, kernel, iterations=2)
reconstructed = cv2.dilate(marker, kernel, iterations=2)

# Pravljenje maske bakarnog novčića
mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
_, contours, _ = cv2.findContours(reconstructed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area >= 0.9 * img.shape[0] * img.shape[1] * 0.01 and area <= 1.1 * img.shape[0] * img.shape[1] * 0.05:
        cv2.drawContours(mask, [contour], 0, 255, -1)

# Cuvanje izlazne maske
cv2.imwrite('coin_mask.png', mask)
