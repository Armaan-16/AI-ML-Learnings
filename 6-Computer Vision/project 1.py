import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import urllib.request
import os

# --- 1. Auto-Download the Required Google Model ---
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading the new MediaPipe model (this only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# --- 2. EAR Calculation Logic ---
def calculate_ear(eye_indices, landmarks):
    p = []
    # Extract the X and Y coordinates for the eye points
    for idx in eye_indices:
        landmark = landmarks[idx]
        p.append((landmark.x, landmark.y))
        
    v1 = math.dist(p[1], p[5])
    v2 = math.dist(p[2], p[4])
    h = math.dist(p[0], p[3])
    
    return (v1 + v2) / (2.0 * h)

# Standard eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.22      
CONSECUTIVE_FRAMES = 25   
frame_counter = 0

# --- 3. Initialize the New Tasks API ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE # Image mode is best for simple webcam loops
)

print("Starting camera... Press 'q' to quit.")
cap = cv2.VideoCapture(0)

# Create the landmarker object
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip for selfie-view and convert to RGB
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Format the frame for the new MediaPipe API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces
        result = landmarker.detect(mp_image)
        
        status_text = "Tracking Face..."
        status_color = (0, 255, 0)

        # If a face is found
        if result.face_landmarks:
            # Get the landmarks for the first face detected
            landmarks = result.face_landmarks[0]
            
            left_ear = calculate_ear(LEFT_EYE, landmarks)
            right_ear = calculate_ear(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    status_text = "ALARM: DROWSY DRIVER!"
                    status_color = (0, 0, 255)
                else:
                    status_text = "Eyes Closed..."
                    status_color = (0, 255, 255)
            else:
                frame_counter = 0
                status_text = "Driver Alert"
                
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            status_text = "No Face Detected"
            status_color = (0, 0, 255)
            frame_counter = 0

        # Draw the main status
        cv2.putText(frame, status_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)
        
        cv2.imshow('Python 3.13 Drowsiness Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()