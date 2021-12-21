import os
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#Define Classes in Dictionary and Load Created Model Previously
classes = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25
}

model = tf.keras.models.load_model('model_SIBI.h5')
model.summary()

import cv2
import mediapipe as mp
import time
 
cap = cv2.VideoCapture(0)
 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
 
pTime = 0
cTime = 0

#Define the Input for the Inference
# Wrist Hand
wristX = 0
wristY = 0
wristZ = 0
                
# Thumb Finger
thumb_CmcX = 0
thumb_CmcY = 0
thumb_CmcZ = 0

thumb_McpX = 0
thumb_McpY = 0
thumb_McpZ = 0
                
thumb_IpX = 0
thumb_IpY = 0
thumb_IpZ = 0
                
thumb_TipX = 0
thumb_TipY = 0
thumb_TipZ = 0

# Index Finger
index_McpX = 0
index_McpY = 0
index_McpZ = 0
                
index_PipX = 0
index_PipY = 0
index_PipZ = 0
                
index_DipX = 0
index_DipY = 0
index_DipZ = 0
                
index_TipX = 0
index_TipY = 0
index_TipZ = 0

# Middle Finger
middle_McpX = 0
middle_McpY = 0
middle_McpZ = 0
                
middle_PipX = 0
middle_PipY = 0
middle_PipZ = 0
                
middle_DipX = 0
middle_DipY = 0
middle_DipZ = 0
                
middle_TipX = 0
middle_TipY = 0
middle_TipZ = 0

# Ring Finger
ring_McpX = 0
ring_McpY = 0
ring_McpZ = 0
                
ring_PipX = 0
ring_PipY = 0
ring_PipZ = 0
                
ring_DipX = 0
ring_DipY = 0
ring_DipZ = 0
                
ring_TipX = 0
ring_TipY = 0
ring_TipZ = 0

# Pinky Finger
pinky_McpX = 0
pinky_McpY = 0
pinky_McpZ = 0
                
pinky_PipX = 0
pinky_PipY = 0
pinky_PipZ = 0
                
pinky_DipX = 0
pinky_DipY = 0
pinky_DipZ = 0
                
pinky_TipX = 0
pinky_TipY = 0
pinky_TipZ = 0

while True:
    success, img = cap.read()
    results = hands.process(cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 1))
    image_height, image_width, _ = img.shape
    # print(results.multi_hand_landmarks)
    img = cv2.flip(img.copy(), 1)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Wrist Hand /  Pergelangan Tangan
            wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
            wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
            wristZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

            # Thumb Finger / Ibu Jari
            thumb_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
            thumb_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
            thumb_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z
                
            thumb_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
            thumb_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
            thumb_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z
                
            thumb_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
            thumb_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
            thumb_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z
                
            thumb_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
            thumb_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
            thumb_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

            # Index Finger / Jari Telunjuk
            index_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
            index_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
            index_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z
                
            index_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
            index_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
            index_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z
                
            index_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
            index_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
            index_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z
                
            index_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
            index_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
            index_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

            # Middle Finger / Jari Tengah
            middle_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
            middle_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
            middle_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z
                
            middle_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
            middle_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
            middle_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z
                
            middle_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
            middle_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
            middle_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z
                
            middle_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
            middle_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
            middle_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

            # Ring Finger / Jari Cincin
            ring_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
            ring_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
            ring_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z
                
            ring_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
            ring_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
            ring_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z
                
            ring_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
            ring_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
            ring_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z
                
            ring_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
            ring_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
            ring_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

            # Pinky Finger / Jari Kelingking
            pinky_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
            pinky_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
            pinky_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z
                
            pinky_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
            pinky_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
            pinky_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z
                
            pinky_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
            pinky_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
            pinky_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z
                
            pinky_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
            pinky_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
            pinky_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

            # Draw the Skeleton
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    
    input_IMG = [[[wristX], [wristY], [wristZ],
                  [thumb_CmcX], [thumb_CmcY], [thumb_CmcZ],
                  [thumb_McpX], [thumb_McpY], [thumb_McpZ], 
                  [thumb_IpX], [thumb_IpY], [thumb_IpZ], 
                  [thumb_TipX], [thumb_TipY], [thumb_TipZ],
                  [index_McpX], [index_McpY], [index_McpZ],
                  [index_PipX], [index_PipY], [index_PipZ],
                  [index_DipX], [index_DipY], [index_DipZ],
                  [index_TipX], [index_TipY], [index_TipZ],
                  [middle_McpX], [middle_McpY], [middle_McpZ],
                  [middle_PipX], [middle_PipY], [middle_PipZ],
                  [middle_DipX], [middle_DipY], [middle_DipZ],
                  [middle_TipX], [middle_TipY], [middle_TipZ],
                  [ring_McpX], [ring_McpY], [ring_McpZ],
                  [ring_PipX], [ring_PipY], [ring_PipZ], 
                  [ring_DipX], [ring_DipY], [ring_DipZ], 
                  [ring_TipX], [ring_TipY], [ring_TipZ],
                  [pinky_McpX], [pinky_McpY], [pinky_McpZ], 
                  [pinky_PipX], [pinky_PipY], [pinky_PipZ],
                  [pinky_DipX], [pinky_DipY], [pinky_DipZ], 
                  [pinky_TipX], [pinky_TipY], [pinky_TipZ]]]
    IMG_array = np.array(input_IMG)
    IMG_array.shape
    predictions = model.predict_classes(IMG_array)
    for alphabets, values in classes.items():
        if values == predictions[0] :
            text_prediction = alphabets
    cv2.putText(img, str(text_prediction), (90, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()