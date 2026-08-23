import cv2
import torch
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model2 import SignLanguageTransformer
from collections import deque, Counter

# ── Config ───────────────────────────────────────────────────
FIXED_LEN   = 100
FEAT_DIM    = 291
NUM_CLASSES = 52
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
POSE_DIMS = 5
HAND_DIMS = 3

# ── Load sign labels ─────────────────────────────────────────
with open(r"E:\Sign_Train_First\class_mapping.json") as f:
    sign_to_idx = json.load(f)
idx_to_sign = {v: k for k, v in sign_to_idx.items()}
print(f"✅ Loaded {len(sign_to_idx)} signs")

# ── Load model ───────────────────────────────────────────────
model = SignLanguageTransformer(
    input_dim   = FEAT_DIM,
    num_classes = NUM_CLASSES,
    max_seq_len = FIXED_LEN
).to(DEVICE)
model.load_state_dict(torch.load(
    r"E:\Sign_Train_First\best_model2.pth",
    map_location=DEVICE,
    weights_only=False
))
model.eval()
print("✅ Model loaded!")

# ── Setup MediaPipe new API ───────────────────────────────────
hand_base    = python.BaseOptions(
    model_asset_path=r"E:\Sign_Train_First\hand_landmarker.task")
hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

pose_base    = python.BaseOptions(
    model_asset_path=r"E:\Sign_Train_First\pose_landmarker.task")
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base,
    running_mode=vision.RunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
print("✅ MediaPipe loaded!")

# ── Landmark extraction ───────────────────────────────────────
def extract_landmarks(frame, timestamp):
    """
    Extract 291-dim feature — SAME ORDER as training:
    pose(165) + left_hand(63) + right_hand(63) = 291
    
    Note: training used visibility+presence from mp.solutions.holistic
    We simulate this with new API (visibility available, presence=0)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp)

    # ── Pose: 33 × 5 = 165 (x,y,z,visibility,presence) ──────
    pose = np.zeros((NUM_POSE_LANDMARKS, POSE_DIMS), dtype=np.float32)
    if pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            pose[i] = [
                lm.x, lm.y, lm.z,
                getattr(lm, 'visibility', 0.0),
                getattr(lm, 'presence', 0.0)
            ]

    # ── Hands: 21 × 3 = 63 each ──────────────────────────────
    left_hand  = np.zeros((NUM_HAND_LANDMARKS, HAND_DIMS), dtype=np.float32)
    right_hand = np.zeros((NUM_HAND_LANDMARKS, HAND_DIMS), dtype=np.float32)

    if hand_result.hand_landmarks:
        for i, hand in enumerate(hand_result.hand_landmarks):
            handedness = hand_result.handedness[i][0].display_name
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand],
                              dtype=np.float32)
            if handedness == "Left":
                left_hand = coords
            else:
                right_hand = coords

    # ── Concatenate in SAME ORDER as training ─────────────────
    feature = np.concatenate([
        pose.flatten(),        # 165 FIRST
        left_hand.flatten(),   # 63  SECOND
        right_hand.flatten()   # 63  THIRD
    ])

    hand_detected = len(hand_result.hand_landmarks) > 0

    return feature, hand_result, pose_result, hand_detected

# ── Predict ───────────────────────────────────────────────────
def predict(buffer):
    actual_len = min(len(buffer), FIXED_LEN)
    seq = np.array(list(buffer)[-actual_len:], dtype=np.float32)

    if actual_len < FIXED_LEN:
        pad = np.zeros((FIXED_LEN - actual_len, FEAT_DIM), dtype=np.float32)
        seq = np.vstack([seq, pad])

    seq_t    = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    length_t = torch.LongTensor([actual_len]).to(DEVICE)

    with torch.no_grad():
        output   = model(seq_t, length_t)
        probs    = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        top3_probs, top3_idx = torch.topk(probs, 3, dim=1)

    top3 = [(idx_to_sign[top3_idx[0][i].item()],
             top3_probs[0][i].item() * 100)
            for i in range(3)]

    return idx_to_sign[pred_idx.item()], conf.item() * 100, top3

# ── Draw landmarks ────────────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_landmarks(frame, hand_result, pose_result):
    h, w, _ = frame.shape

    if pose_result.pose_landmarks:
        for lm in pose_result.pose_landmarks[0]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            for s, e in HAND_CONNECTIONS:
                x1,y1 = int(hand[s].x*w), int(hand[s].y*h)
                x2,y2 = int(hand[e].x*w), int(hand[e].y*h)
                cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)

# ── Main Loop ─────────────────────────────────────────────────
cap          = cv2.VideoCapture(0)
buffer       = deque(maxlen=FIXED_LEN)
predicted    = "Show a sign..."
confidence   = 0.0
top3_preds   = []
frame_count  = 0
timestamp    = 0
pred_history = deque(maxlen=5)
MIN_FRAMES   = 30

print("✅ Camera opened!")
print("Press SPACE to reset | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp   += 1

    landmarks, hand_result, pose_result, hand_detected = \
        extract_landmarks(frame, timestamp)

    if hand_detected:
        buffer.append(landmarks)
    else:
        if len(buffer) > 0:
            buffer.popleft()

    draw_landmarks(frame, hand_result, pose_result)

    if (frame_count % 25 == 0 and
        len(buffer) >= MIN_FRAMES and
        hand_detected):
        sign, conf, top3 = predict(buffer)
        pred_history.append(sign)
        predicted  = Counter(pred_history).most_common(1)[0][0]
        confidence = conf
        top3_preds = top3

    elif not hand_detected and len(buffer) < MIN_FRAMES:
        predicted  = "Show a sign..."
        confidence = 0.0
        top3_preds = []
        pred_history.clear()

    h, w, _ = frame.shape

    cv2.rectangle(frame, (0,0), (340,200), (0,0,0), -1)
    cv2.rectangle(frame, (0,0), (340,200), (50,50,50), 2)

    cv2.putText(frame, predicted,
               (10,55), cv2.FONT_HERSHEY_SIMPLEX,
               1.5, (0,255,0), 3)

    cv2.putText(frame, f"Conf: {confidence:.1f}%",
               (10,90), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0,255,255), 2)

    if top3_preds:
        cv2.putText(frame, "Top 3:",
                   (10,115), cv2.FONT_HERSHEY_SIMPLEX,
                   0.55, (200,200,200), 1)
        for i, (s, p) in enumerate(top3_preds):
            cv2.putText(frame, f"  {i+1}. {s} ({p:.0f}%)",
                       (10, 135+i*20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (180,180,180), 1)

    hand_color = (0,255,0) if hand_detected else (0,0,255)
    cv2.putText(frame,
               f"Hand:{'ON' if hand_detected else 'OFF'} | Buf:{len(buffer)}/{FIXED_LEN}",
               (10, h-10),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.55, hand_color, 1)

    cv2.imshow("NSL Sign Recognition v2 — Q:quit SPACE:reset", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        buffer.clear()
        pred_history.clear()
        predicted  = "Show a sign..."
        confidence = 0.0
        top3_preds = []
        print("🔄 Reset!")

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
pose_landmarker.close()
print("✅ Done")