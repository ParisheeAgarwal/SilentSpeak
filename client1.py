import socket
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Define the server's IP address and port
SERVER_HOST = '127.0.0.1'  # Loopback address
SERVER_PORT = 12345

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
classifier = Classifier("keras_model.h5", "labels.txt")
offset = 20
imgSize = 300

counter = 0
last_index = None
a = []

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'z_unknown']

# Create a socket object
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    # Connect to the server
    client_socket.connect((SERVER_HOST, SERVER_PORT))
    
    send_data = True  # Flag to control sending data
    receive_data = False  # Flag to control receiving data

    while True:
        if receive_data:
            # Receive data from the server
            received_data = client_socket.recv(1024)
            if received_data:
                print("Received data from server:", received_data.decode())

        if send_data:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                if w > 0 and h > 0:  # Check if width and height are greater than 0
                    imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
                    if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if imgCrop is not empty
                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 225
                        
                        aspectRatio = h / w

                        if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            imgWhite[:, :wCal] = imgResize

                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            imgWhite[:hCal, :] = imgResize

                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

                        if index == last_index:
                            counter += 1
                        else:
                            counter = 0

                        if counter >= 15:
                            print("Letter:", labels[index])
                            a.append(labels[index])
                            counter = 0

                        last_index = index

                        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
                    
            cv2.imshow("Image", imgOutput)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                # Send the updated list 'a' to the server if sending is allowed
                if send_data:
                    client_socket.sendall(str(a).encode())
                    print('Data sent to server:', a)
                    a = []  # Clear the list after sending data

            elif key & 0xFF == ord('s'):
                # Toggle sending data flag
                send_data = not send_data

            elif key & 0xFF == ord('r'):
                # Toggle receiving data flag
                receive_data = not receive_data

            elif key == 27:  # Press Esc to exit
                break

print(' '.join(a))
