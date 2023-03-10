import cv2
import numpy as np

# Učitavanje slike
img = cv2.imread('coins.png')

# Konverzija slike u grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Segmentacija novčića
th = 150
max_val = 255
_, thresh = cv2.threshold(gray, th, max_val, cv2.THRESH_BINARY_INV)

# Morfološke operacije za popunjavanje rupa
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Filtriranje novčića po veličini
min_area = 0.9 * np.pi * (60/2)**2  # Minimum površina novčića
max_area = 1.1 * np.pi * (80/2)**2  # Maksimum površina novčića
cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(img)
for cnt in cnts:
    area = cv2.contourArea(cnt)
    if min_area < area < max_area:
        cv2.drawContours(mask, [cnt], 0, (255,255,255), -1)

# Konverzija slike u HSV prostor boja
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Segmentacija markera
th = 20
_, sat = cv2.threshold(hsv[:,:,1], th, max_val, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
sat = cv2.morphologyEx(sat, cv2.MORPH_CLOSE, kernel)

# Izdvajanje bakarnog novčića
marker = np.zeros_like(sat)
marker[70:120, 260:320] = 255  # Ručno označavanje bakarnog novčića
dst = cv2.dilate(marker, kernel, iterations=3)
dst = cv2.min(dst, sat)
markers = cv2.connectedComponents(dst)[1]
markers = markers + 1
markers[mask == 0] = 0
mask = cv2.watershed(img, markers)
mask[mask == 2] = 0

# Ispis izlazne maske
cv2.imwrite('coin_mask.png', mask)

