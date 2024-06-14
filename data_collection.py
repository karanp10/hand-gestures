# Install the necessary packages
import cv2
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp
import os
import csv
import joblib
#import testing

# Visualization utilities
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

model = joblib.load('gesture_recognition_model.pkl')

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        landmarks_array = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks]
        gesture_name = "right_hand_indexfinger_pointingdown"
        save_gesture_data(landmarks_array, gesture_name, handedness[0].category_name)
        #testing.process_landmarks(landmarks_array)
        #print('Hand landmarks:', landmarks_array)

        #save_landmark_data(landmarks_array, handedness[0].category_name)
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

def save_landmark_data(landmarks, handedness):
    file = os.path.isfile('right_hand_fist.csv')
    with open('right_hand_fist.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file:
            header = ['handedness'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)
        row = [handedness] + [coord for landmark in landmarks for coord in landmark]
        writer.writerow(row) 

def save_gesture_data(landmarks, gesture_name, handedness):
    file = os.path.isfile('gesture_data.csv')
    with open('right_hand_fist.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file:
            header = ['gesture_name', 'handedness'] + [f'landmark_{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)
        row = [gesture_name, handedness] + [coord for landmark in landmarks for coord in landmark]
        writer.writerow(row)




def capture():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a MediaPipe image object
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect hand landmarks
        detection_result = detector.detect(image)

        # Annotate the frame with hand landmarks
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

        # Convert the frame back to BGR for OpenCV
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Display the annotated frame
        cv2.imshow('Hand Landmark Detection', bgr_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()

