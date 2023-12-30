import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Change maxHands to 2

imgSz = 500  # Increase the size for a larger frame
topFrameSize = 100  # Increase the size of the top frame
sideFrameSize = 50

folder = "img/lots of love"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgSz, imgSz, 3), np.uint8) * 255
    imgCrop = np.ones((imgSz, imgSz, 3), np.uint8) * 255  # Initialize imgCrop

    for hand in hands:
        x, y, w, h = hand['bbox']
        handRegion = img[y - topFrameSize:y + h + topFrameSize, x - sideFrameSize:x + w + sideFrameSize]
        handRegion = cv2.resize(handRegion, (imgSz, imgSz))
        imgCrop = cv2.bitwise_and(imgCrop, handRegion)  # Accumulate both hands

    imgWhite[0:imgSz, 0:imgSz] = imgCrop

    cv2.imshow("ImageCrop", imgCrop)
    cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
