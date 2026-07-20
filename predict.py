import cv2
import torch
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import SignLanguageTransformer
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sign labels
with open("sign_to_idx.json") as f:
    sign_to_idx = json.load(f)
idx_to_sign = {v: k for k, v in sign_to_idx.items()}

# Load model
model = SignLanguageTransformer().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Setup MediaPipe
hand_base = python.BaseOptions(model_asset_path="hand_landmarker.task")
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

pose_base = python.BaseOptions(model_asset_path="pose_landmarker.task")
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.3,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

def extract_live_landmarks(frame, hand_landmarker, pose_landmarker, timestamp):
    """Extract landmarks from one live frame"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp)

    landmarks = []

    left_hand  = [0.0] * 63
    right_hand = [0.0] * 63

    if hand_result.hand_landmarks:
        for i, hand in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].display_name
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            if handedness == "Left":
                left_hand = coords
            else:
                right_hand = coords

    landmarks.extend(left_hand)
    landmarks.extend(right_hand)

    pose = [0.0] * 99
    if pose_result.pose_landmarks:
        coords = []
        for lm in pose_result.pose_landmarks[0]:
            coords.extend([lm.x, lm.y, lm.z])
        pose = coords

    landmarks.extend(pose)

    return np.array(landmarks), hand_result, pose_result

# ── Main Loop ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("✅ Camera opened")
print("Perform a sign for 25 frames — prediction appears automatically")
print("Press Q to quit")

# Buffer to collect 25 frames
frame_buffer = deque(maxlen=25)
timestamp    = 0
predicted    = "Waiting for sign..."
confidence   = 0.0

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp += 1

    # Extract landmarks
    landmarks, hand_result, pose_result = extract_live_landmarks(
        frame, hand_landmarker, pose_landmarker, timestamp
    )

    # Add to buffer
    frame_buffer.append(landmarks)

    h, w, _ = frame.shape

    # Draw hand landmarks
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            for s, e in CONNECTIONS:
                x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
                x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
                cv2.line(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)

    # Draw pose landmarks
    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

    # Predict when buffer is full (every 25 frames)
    if len(frame_buffer) == 25 and timestamp % 25 == 0:
        sequence = np.array(list(frame_buffer))
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            probs  = torch.softmax(output, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            predicted  = idx_to_sign[pred_idx.item()]
            confidence = conf.item() * 100

    # Display
    cv2.putText(frame,
               f"Sign: {predicted}",
               (10, 40),
               cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 255, 0), 2)

    cv2.putText(frame,
               f"Conf: {confidence:.1f}%",
               (10, 80),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0, 255, 255), 2)

    cv2.putText(frame,
               f"Frames: {len(frame_buffer)}/25",
               (10, 120),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (255, 255, 255), 2)

    cv2.imshow("NSL Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
pose_landmarker.close()
print("✅ Done")