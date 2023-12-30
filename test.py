import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import time
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

imgSz = 500  # Increase the size for a larger frame
topFrameSize = 100  # Increase the size of the top frame
sideFrameSize = 50

folder = "img/C"
counter = 0

labels = ["hello","I need help", "lots of love"]

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgSz, imgSz, 3), np.uint8) * 255
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - topFrameSize:y + h + topFrameSize, x - sideFrameSize:x + w + sideFrameSize]
        imgCrop = cv2.resize(imgCrop, (imgSz, imgSz))
        imgWhite[0:imgSz, 0:imgSz] = imgCrop

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        prediction, index = classifier.getPrediction(img)
        print(prediction, index)

    cv2.imshow("Image", img)
    cv2.waitKey(1)