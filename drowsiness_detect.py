from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time
import sys

# Initialize sound mixer
mixer.init()
try:
    mixer.music.load("music.wav")
except Exception as e:
    print(f"Error loading sound file: {e}")
    sys.exit(1)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants
thresh = 0.25
frame_check = 20

# Initialize dlib's face detector and landmark predictor
detect = dlib.get_frontal_face_detector()
try:
    predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print(f"Error loading shape predictor: {e}")
    sys.exit(1)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Trying alternative indices...")
    # Try different camera indices
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Found camera at index {i}")
            break
    else:
        print("Error: No camera found at indices 0-2")
        sys.exit(1)

# Allow camera to warm up
time.sleep(2.0)

flag = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Could not read frame. Trying again...")
        time.sleep(0.1)
        continue

    try:
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    try:
                        mixer.music.play()
                    except:
                        pass  # Skip sound if there's an error
            else:
                flag = 0
        
        cv2.imshow("Drowsiness Detection", frame)
        
    except Exception as e:
        print(f"Processing error: {e}")
        continue

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()         uiyuoyioihl