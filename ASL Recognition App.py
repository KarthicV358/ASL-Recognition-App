import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

# Load your pre-trained model
model = tf.keras.models.load_model('asl_finetuned_model2.h5')

# Initialize MediaPipe Hand detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Custom drawing style for the connections (green) and landmarks (red)
connection_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)  # Green connections
landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)  # Red landmarks

# Parameters
imgSize = 300  # The size of the hand landmark images in your dataset
input_size = 224  # Size expected by MobileNetV2 or your model
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # Assuming 26 alphabet classes

# Initialize webcam
cap = cv2.VideoCapture(0)

# Real-time detection loop
while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image to RGB (required for MediaPipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Create a blank white image (as in your dataset)
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Draw landmarks and connections with the style used in training data
            mp_drawing.draw_landmarks(
                imgWhite, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

            # Resize the image to the input size expected by the model (224x224)
            img_resized = cv2.resize(imgWhite, (input_size, input_size))
            img_resized = img_resized / 255.0  # Normalize pixel values
            img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

            # Predict the corresponding alphabet
            predictions = model.predict(img_resized)
            predicted_index = np.argmax(predictions)
            predicted_label = labels[predicted_index]

            # Display the prediction on the webcam feed
            cv2.putText(img, f'Prediction: {predicted_label}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            # Optionally, display the hand landmark image as well
            cv2.imshow("Landmark Image", imgWhite)

    # Show the webcam feed with prediction
    cv2.imshow("ASL Detection", img)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
