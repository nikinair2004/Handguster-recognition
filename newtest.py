import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3  # Text-to-speech library

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speech rate

cap = cv2.VideoCapture(0)  # Opens the default camera
detector = HandDetector(maxHands=1)  # Detecting a maximum of one hand at a time
# Loads a pre-trained TensorFlow model and the gesture corresponding labels
classifier = Classifier("/Users/nikitanair/PycharmProjects/pythonProject3/Model2/keras_model.h5", "/Users/nikitanair/PycharmProjects/pythonProject3/Model2/labels.txt")
offset = 20  # Adds padding around the hands bounding box for better cropping
imgSize = 300  # Use size of image for classification

#folder = "C:/NFThandrec/pythonProject/data"  # Path where data must be stored
labels = ["down", "fist", "okay", "palm", "thumbs", "up"]  # List of gestures

while True:
    success, img = cap.read()  # Continuously read the feed
    imgOutput = img.copy()  # Copy of image
    hands, img = detector.findHands(img)  # Detects the hands in the feed
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # The boundary box provides the position
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # A 300x300 white image as background
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]  # Crop the hand part
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:  # If height is more, adjust height to 300 px and width proportionally
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Draw bounding box and display text and accuracy on image
        # The prediction array contains confidence scores for all labels
        accuracy = prediction[index] * 100  # Convert accuracy to percentage
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 200, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        # displays the label's name along with its corresponding accuracy ex:- thumbs 95.23%
        cv2.putText(imgOutput, f'{labels[index]} {accuracy:.2f}%', (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        # Convert detected gesture to speech
        # text-to-speech engine announces the gesture along with its accuracy
        gesture_text = labels[index]
        engine.say(f"{gesture_text} with {accuracy:.2f} percent accuracy")  # ex:-thumbs with 95.23 percent accuracy
        engine.runAndWait()

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)