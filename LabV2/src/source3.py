import cv2
import numpy as np

# Učitavanje slike novčića
img = cv2.imread('coins.jpg')

# Pretvaranje u grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Prag segmentacije (threshold) - pronađen ručno
threshold_value = 130

# Primena thresholda za segmentaciju novčića
ret, mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

# Popunjavanje rupa unutar novčića morfološkom operacijom zatvaranja
kernel = np.ones((5,5), np.uint8)
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Pretvaranje slike u HSV prostor boja
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Segmentacija markera (za morfološku rekonstrukciju) na Saturation kanalu
marker_threshold_value = 80
marker_mask = hsv[:,:,1] > marker_threshold_value

# Filtriranje nepotrebnih piksela iz markera morfološkom operacijom otvaranja
kernel2 = np.ones((3,3), np.uint8)
opened_marker_mask = cv2.morphologyEx(marker_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel2)

# Izdvajanje bakarnog novčića morfološkom rekonstrukcijom
marker = np.zeros_like(mask)
marker[10:-10, 10:-10] = opened_marker_mask[10:-10, 10:-10] # marker na pozicijama gde je detektovan bakarni novčić
reconstructed = cv2.morphologyEx(closed_mask, cv2.MORPH_RECONSTRUCT, kernel, marker.astype(np.uint8))

# Crtanje pronađenih kontura oko novčića
contours, _ = cv2.findContours(reconstructed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Crtanje pravougaonika oko bakarnog novčića
x, y, w, h = cv2.boundingRect(marker.astype(np.uint8))
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Čuvanje rezultujuće maske i prikazivanje rezultata
cv2.imwrite('coin_mask.png', reconstructed)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
