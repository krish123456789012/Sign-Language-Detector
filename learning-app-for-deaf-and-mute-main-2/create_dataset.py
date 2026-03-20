import os  # Import the os module for interacting with the operating system
import pickle  # Import the pickle module for serializing and deserializing data

import mediapipe as mp  # Import the mediapipe library for hand tracking
import cv2  # Import the OpenCV library for image processing
import matplotlib.pyplot as plt  # Import matplotlib for plotting (not used in this code)

# Initialize Mediapipe's hand solutions
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Create a Mediapipe Hands object with specified parameters
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory where the dataset images are stored
DATA_DIR = './data'

# Lists to store the processed data and corresponding labels
data = []
labels = []

# Iterate through the directories inside DATA_DIR (each directory represents a class)
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Check if the current path is a directory
    if os.path.isdir(dir_path):
        # Iterate through each image in the class directory
        for img_path in os.listdir(dir_path):
            data_aux = []  # Temporary list to store processed hand landmarks
            x_ = []  # Temporary list to store x-coordinates of landmarks
            y_ = []  # Temporary list to store y-coordinates of landmarks

            # Read the image using OpenCV
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            # Convert the image from BGR (OpenCV format) to RGB (Mediapipe format)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to detect hand landmarks
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                # Iterate through detected hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    # Collect x and y coordinates of each landmark
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Normalize the landmarks relative to the minimum x and y values
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))  # Normalize x-coordinate
                        data_aux.append(y - min(y_))  # Normalize y-coordinate

                # Append the normalized data and corresponding label (class) to the lists
                data.append(data_aux)
                labels.append(dir_)

# Save the processed data and labels to a pickle file for later use
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
