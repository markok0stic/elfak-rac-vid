import cv2
import numpy as np

# učitavanje slike novčića
img = cv2.imread('coins.jpg')

# pretvaranje slike u grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# thresholding za segmentaciju novčića
# ručno se podešava prag (u ovom slučaju 100)
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# popunjavanje rupa unutar novčića
kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# filtriranje viškova
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

# pretvaranje slike u HSV prostor boja
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# segmentacija markera na Saturation kanalu
# ručno se podešava prag (u ovom slučaju 50)
_, marker = cv2.threshold(hsv[:,:,1], 50, 255, cv2.THRESH_BINARY)

# filtriranje nepotrebnih piksela
marker_filtered = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)

# morfološka rekonstrukcija za izdvajanje bakarnog novčića
# rezultat je izlazna maska bakarnog novčića
mask = cv2.dilate(marker_filtered, kernel, iterations=3)
mask = cv2.subtract(mask, opening)

# prikazivanje rezultata
cv2.imshow('Original image', img)
cv2.imshow('Thresholded image', thresh)
cv2.imshow('Closed image', closing)
cv2.imshow('Opened image', opening)
cv2.imshow('Marker', marker)
cv2.imshow('Filtered marker', marker_filtered)
cv2.imshow('Coin mask', mask)
cv2.waitKey(0)

# čuvanje izlazne maske
cv2.imwrite('coin_mask.png', mask)
