import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Function to resize image to (224, 224, 3) and convert to grayscale
def resize_and_convert_to_gray(img):
    resized_img = cv2.resize(img, (224, 224))
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    gray_img = np.stack((gray_img,) * 3, axis=-1)  # Stack grayscale channel to create a 3-channel image
    return gray_img

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 300

counter = 0
last_index = None
a = []

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'z_unknown']

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Resize the cropped image and convert to grayscale
        imgResize = resize_and_convert_to_gray(imgCrop)

        # Get prediction from classifier
        prediction, index = classifier.getPrediction(imgResize, draw=False)

        if index == last_index:
            counter += 1
        else:
            counter = 0

        if counter >= 10:
            print("Letter:", labels[index])
            a.append(labels[index])
            counter = 0

        last_index = index

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(' '.join(a))

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
