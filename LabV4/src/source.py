import numpy as np
import cv2 as cv
import imutils

img = cv.imread('canvas.png')
(partW, partH) = (180, 180)

# manually inputed coords
img = img[120:841, 401:1842]
imgProba = img.copy()

# load the class labels from disk
rows = open('synset_words.txt').read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
# load our serialized model from disk
net = cv.dnn.readNetFromCaffe('bvlc_googlenet.prototxt', 'bvlc_googlenet.caffemodel')

i = 0
stepSizeX = 180
stepSizeY = 180
scale = 2

while img.shape[0] >= partW and img.shape[1] >= partH:
    for y in range(0, img.shape[0], stepSizeY):
        for x in range(0, img.shape[1], stepSizeX):
            partImg = img[y:y+partH, x:x+partW]

            clone = partImg.copy()
            blob = cv.dnn.blobFromImage(partImg, 1, (224, 224), (104, 117, 123))

            # set the blob as input to the network and perform a forward-pass to
            # obtain our output classification
            net.setInput(blob)
            preds = net.forward()

            idxT = (np.argsort(preds[0])[::-1][:5])[0]
            text = ''
            if 'dog' in classes[idxT]:
                text = 'DOG'
            if 'cat' in classes[idxT]:
                text = 'CAT'

            if preds[0][idxT] > 0.9 and text == 'CAT':
                cv.rectangle(imgProba, (x*scale**i, y*scale**i),
                             (x*scale**i + partW*scale**i, y*scale**i + partH*scale**i), (0, 0, 255), 2)
                cv.putText(imgProba, text, (x * scale**i + 5, y * scale**i + 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if preds[0][idxT] > 0.68 and text == 'DOG':
                cv.rectangle(imgProba, (x * scale ** i, y * scale ** i),
                             (x * scale ** i + partW * scale ** i, y*scale**i + partH * scale ** i), (0, 255, 255), 2)
                cv.putText(imgProba, text, (x * scale**i + 5, y * scale**i + 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    w = int(img.shape[1] / scale)
    img = imutils.resize(img, width=w)
    i = i + 1

cv.imshow('Output', imgProba)
cv.imwrite('output.png', imgProba)
cv.waitKey()
cv.destroyAllWindows()
