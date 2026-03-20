import os  # Import the os module to interact with the operating system
import cv2  # Import the OpenCV library for computer vision tasks

# Directory where the dataset will be stored
DATA_DIR = './data'

# Create the directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes and the size of the dataset per class
number_of_classes = 3
dataset_size = 100

# Start video capture from the webcam (device index 0)
cap = cv2.VideoCapture(0)

# Loop through each class to collect data
for j in range(number_of_classes):
    # Create a subdirectory for each class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for the user to press "Q" to start collecting images
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)  # Display instructions on the frame
        cv2.imshow('frame', frame)  # Show the frame in a window
        if cv2.waitKey(25) == ord('q'):  # Check if the "Q" key is pressed
            break  # Exit the loop to start capturing images

    # Initialize a counter to keep track of the number of images saved
    counter = 0

    # Capture and save images until the dataset size is reached
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the webcam
        cv2.imshow('frame', frame)  # Show the frame in a window
        cv2.waitKey(25)  # Wait for 25 milliseconds between frames

        # Save the captured frame as an image file in the appropriate class directory
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1  # Increment the counter

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
