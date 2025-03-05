import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize the hand detector with a max of 1 hand
detector = HandDetector(maxHands=1)

# Load the trained model and labels
classifier = Classifier("/Users/nikitanair/PycharmProjects/pythonProject3/Model3/keras_model.h5", "/Users/nikitanair/PycharmProjects/pythonProject3/Model3/labels.txt")

# Constants
offset = 20
imgSize = 300
labels = ["down", "fist", "okay", "palm", "thumbs", "up"]

while True:
    success, img = cap.read()  # Read the webcam frame
    if not success:
        print("Failed to capture image from webcam.")
        break

    imgOutput = img.copy()  # Make a copy of the frame for drawing purposes
    hands, img = detector.findHands(img)  # Detect the hand

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  # Get the bounding box of the hand

        # Print bounding box coordinates for debugging
        print(f"Bounding box coordinates: x={x}, y={y}, w={w}, h={h}")

        # Clamp coordinates to be within image dimensions
        h, w, _ = img.shape
        x = max(0, x - offset)
        y = max(0, y - offset)
        w = min(w, x + w + 2 * offset) - x
        h = min(h, y + h + 2 * offset) - y

        # Crop the hand region from the frame
        imgCrop = img[y:y + h, x:x + w]

        # Check if imgCrop is empty
        if imgCrop.size == 0:
            print("Cropped image is empty. Check bounding box coordinates.")
            continue

        # Create a white background image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Get the aspect ratio of the hand bounding box
        aspectRatio = h / w if w != 0 else 0  # Prevent division by zero

        # Resize the image based on the aspect ratio
        if aspectRatio > 1:  # Height > Width
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:  # Width >= Height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Convert the image to grayscale and check if it is empty before reshaping
        imgWhiteGray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)

        # Check if the grayscale image has the correct shape
        if imgWhiteGray.size == 0:
            print("White background grayscale image is empty.")
            continue

        imgWhiteGray = imgWhiteGray.reshape((1, imgSize, imgSize, 1))

        # Get the model's prediction
        try:
            prediction, index = classifier.getPrediction(imgWhiteGray, draw=False)
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue

        # Display the prediction and bounding box on the output image
        cv2.rectangle(imgOutput, (x, y - 50),
                      (x + 90, y - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x, y),
                      (x + w, y + h), (255, 0, 255), 4)

        # Display the cropped and white background images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Show the final output with the bounding box and prediction
    cv2.imshow("Image", imgOutput)

    # Wait for the "q" key to be pressed to break the loop and exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
