import cv2
import torch
import numpy as np
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model2 import SignLanguageTransformer
from collections import deque, Counter
import clip
from PIL import Image
import time

# ── Config ───────────────────────────────────────────────────
FIXED_LEN   = 100
FEAT_DIM    = 291
NUM_CLASSES = 52
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21
POSE_DIMS = 5
HAND_DIMS = 3

# ── Dynamic sliding window config ────────────────────────────
MIN_FRAMES     = 20      # minimum frames before predicting
MAX_FRAMES     = 80      # force commit at this window size
CHECK_INTERVAL = 5       # check confidence every N frames
START_THRESH   = 0.90    # confidence threshold at MIN_FRAMES
END_THRESH     = 0.55    # confidence threshold at MAX_FRAMES

# ── Motion / pause detection config ──────────────────────────
VELOCITY_THRESH    = 0.004  # minimum hand motion to count as signing
WORD_PAUSE_SEC     = 0.4    # stillness = end of one word
SENTENCE_PAUSE_SEC = 1.0    # stillness = end of sentence → clear screen
DEBUG_VELOCITY     = False  # set True to print velocity values

# ── Sentence log ──────────────────────────────────────────────
SENTENCE_LOG_PATH = "detected_sentences.txt"

# ── Load sign labels ─────────────────────────────────────────
with open(r"E:\Sign_Train_First\class_mapping.json") as f:
    sign_to_idx = json.load(f)
idx_to_sign = {v: k for k, v in sign_to_idx.items()}
print(f"✅ Loaded {len(sign_to_idx)} signs")

# ── Load Transformer model ────────────────────────────────────
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

# ── Load CLIP ─────────────────────────────────────────────────
print("Loading CLIP...")
model_clip, preprocess_clip = clip.load("ViT-B/32", device=DEVICE)
model_clip.eval()

SCENES = [
    "a hospital room or medical facility",
    "a classroom or school environment",
    "a home living room",
    "an outdoor street or public place",
    "an office or workplace",
    "a kitchen or dining area",
    "a shop or market",
    "a library or study room",
    "a pharmacy or medical shop",
    "a waiting room or reception area",
    "a restaurant or cafe",
    "a community center or gathering place",
]

# Encode text ONCE at startup
text_tokens = clip.tokenize(SCENES).to(DEVICE)
with torch.no_grad():
    text_features = model_clip.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print("✅ CLIP loaded!")

# ── Setup MediaPipe ───────────────────────────────────────────
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
    Extract 291-dim feature vector.
    Order MUST match training: pose(165) + left_hand(63) + right_hand(63)
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp)

    # Pose: 33 × 5 = 165
    pose = np.zeros((NUM_POSE_LANDMARKS, POSE_DIMS), dtype=np.float32)
    if pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            pose[i] = [
                lm.x, lm.y, lm.z,
                getattr(lm, 'visibility', 0.0),
                getattr(lm, 'presence', 0.0)
            ]

    # Hands: 21 × 3 = 63 each
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

    feature = np.concatenate([
        pose.flatten(),        # 165 — FIRST (matches training)
        left_hand.flatten(),   # 63  — SECOND
        right_hand.flatten()   # 63  — THIRD
    ])

    hand_detected = len(hand_result.hand_landmarks) > 0
    return feature, hand_result, pose_result, hand_detected


# ── Transformer prediction ────────────────────────────────────
def predict(buffer):
    actual_len = min(len(buffer), FIXED_LEN)
    seq = np.array(list(buffer)[-actual_len:], dtype=np.float32)

    if actual_len < FIXED_LEN:
        pad = np.zeros((FIXED_LEN - actual_len, FEAT_DIM), dtype=np.float32)
        seq = np.vstack([seq, pad])

    seq_t    = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
    length_t = torch.LongTensor([actual_len]).to(DEVICE)

    with torch.no_grad():
        output = model(seq_t, length_t)
        probs  = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)
        top3_probs, top3_idx = torch.topk(probs, 3, dim=1)

    top3 = [(idx_to_sign[top3_idx[0][i].item()],
             top3_probs[0][i].item() * 100)
            for i in range(3)]

    return idx_to_sign[pred_idx.item()], conf.item() * 100, top3


# ── CLIP scene detection ──────────────────────────────────────
def detect_scene(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    image_input = preprocess_clip(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        image_features = model_clip.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).squeeze(0)
    probs      = torch.softmax(similarity * 100, dim=0)

    best_idx  = probs.argmax().item()
    best_prob = probs[best_idx].item() * 100

    full_scene  = SCENES[best_idx]
    short_scene = full_scene.split(" or ")[0]
    short_scene = short_scene.replace("a ", "").replace("an ", "").title()

    return short_scene, best_prob


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
                x1, y1 = int(hand[s].x * w), int(hand[s].y * h)
                x2, y2 = int(hand[e].x * w), int(hand[e].y * h)
                cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), 2)


# ── Dynamic threshold ─────────────────────────────────────────
def get_dynamic_threshold(current_len):
    if current_len <= MIN_FRAMES:
        return START_THRESH
    progress = (current_len - MIN_FRAMES) / (MAX_FRAMES - MIN_FRAMES)
    progress = min(progress, 1.0)
    return START_THRESH - progress * (START_THRESH - END_THRESH)


# ── Hand velocity ─────────────────────────────────────────────
def hand_velocity(prev_landmarks, curr_landmarks):
    prev_hands = prev_landmarks[165:]
    curr_hands = curr_landmarks[165:]
    return float(np.mean(np.abs(curr_hands - prev_hands)))


# ── Log sentence ──────────────────────────────────────────────
def log_sentence(words):
    if not words:
        return
    line = " ".join(words)
    with open(SENTENCE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(f"📝 Sentence logged: {line}")


# ── State variables ───────────────────────────────────────────
cap               = cv2.VideoCapture(0)
buffer            = deque(maxlen=FIXED_LEN)
predicted         = "Show a sign..."
confidence        = 0.0
top3_preds        = []
frame_count       = 0
timestamp         = 0
pred_history      = deque(maxlen=5)
current_scene     = "Detecting..."
scene_confidence  = 0.0
last_movement_time = time.time()
sentence_finalized = False

# ── Sentence state ────────────────────────────────────────────
# Stores unique consecutive words
# Same word won't be added twice in a row
# Cleared after SENTENCE_PAUSE_SEC of stillness
sentence_words = []   # list of committed unique words
last_committed = None # last word added to sentence

print("✅ Camera opened!")
print("Controls: SPACE = reset | Q = quit")
print(f"Sentence clears after {SENTENCE_PAUSE_SEC}s stillness")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp   += 1

    # ── CLIP every 30 frames ─────────────────────────────────
    if frame_count % 30 == 0:
        current_scene, scene_confidence = detect_scene(frame)

    # ── Extract landmarks ─────────────────────────────────────
    landmarks, hand_result, pose_result, hand_detected = \
        extract_landmarks(frame, timestamp)

    # Fill or drain buffer based on hand detection
    if hand_detected:
        buffer.append(landmarks)
    else:
        if len(buffer) > 0:
            buffer.popleft()

    # ── Draw skeleton ─────────────────────────────────────────
    draw_landmarks(frame, hand_result, pose_result)

    # ── Motion tracking ───────────────────────────────────────
    if len(buffer) >= 2:
        v = hand_velocity(buffer[-2], buffer[-1])
        if DEBUG_VELOCITY and frame_count % 3 == 0:
            print(f"velocity: {v:.5f}")
        if v > VELOCITY_THRESH:
            last_movement_time = time.time()
            sentence_finalized = False

    stillness       = time.time() - last_movement_time
    word_paused     = stillness >= WORD_PAUSE_SEC
    sentence_paused = stillness >= SENTENCE_PAUSE_SEC

    # ── Sentence clear on long pause ─────────────────────────
    # When still for SENTENCE_PAUSE_SEC → log and clear sentence
    if sentence_paused and sentence_words and not sentence_finalized:
        log_sentence(sentence_words)
        sentence_words  = []
        last_committed  = None
        sentence_finalized = True

    # ── Prediction ────────────────────────────────────────────
    should_check = (frame_count % CHECK_INTERVAL == 0) or word_paused

    if hand_detected and len(buffer) >= MIN_FRAMES and should_check:
        sign, conf, top3 = predict(buffer)
        pred_history.append(sign)
        predicted  = Counter(pred_history).most_common(1)[0][0]
        confidence = conf
        top3_preds = top3

        threshold        = get_dynamic_threshold(len(buffer))
        confident_enough = (conf / 100.0) >= threshold
        window_maxed     = len(buffer) >= MAX_FRAMES

        # Commit word when confident, paused, or window maxed
        if confident_enough or word_paused or window_maxed:
            if word_paused:     reason = "word pause"
            elif confident_enough: reason = "confident"
            else:               reason = "max window"

            print(f"✅ '{predicted}' ({confidence:.1f}%) — {reason}, "
                  f"window={len(buffer)}")

            # ── Add to sentence only if different from last word ──
            # This prevents "ghar ghar ghar" duplicates
            if predicted != last_committed and predicted != "Show a sign...":
                sentence_words.append(predicted)
                last_committed = predicted

                # Keep only last 5 words visible
                if len(sentence_words) > 5:
                    sentence_words.pop(0)

            # Reset buffer for next sign
            buffer.clear()
            pred_history.clear()

    elif not hand_detected and len(buffer) < MIN_FRAMES:
        predicted  = "Show a sign..."
        confidence = 0.0
        top3_preds = []
        pred_history.clear()

    # ══════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════
    h, w, _ = frame.shape

    # ── Top LEFT — current sign prediction ───────────────────
    cv2.rectangle(frame, (0,0), (340, 200), (0,0,0), -1)
    cv2.rectangle(frame, (0,0), (340, 200), (50,50,50), 2)

    cv2.putText(frame, predicted,
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
               1.5, (0,255,0), 3)

    cv2.putText(frame, f"Conf: {confidence:.1f}%",
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (0,255,255), 2)

    if top3_preds:
        cv2.putText(frame, "Top 3:",
                   (10, 115), cv2.FONT_HERSHEY_SIMPLEX,
                   0.55, (200,200,200), 1)
        for i, (s, p) in enumerate(top3_preds):
            cv2.putText(frame, f"  {i+1}. {s} ({p:.0f}%)",
                       (10, 135 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (180,180,180), 1)

    # ── Top RIGHT — CLIP scene ────────────────────────────────
    box_w = 280
    cv2.rectangle(frame, (w-box_w-10, 5), (w-10, 75), (0,0,0), -1)
    cv2.rectangle(frame, (w-box_w-10, 5), (w-10, 75), (50,50,50), 2)

    cv2.putText(frame, "Scene:",
               (w-box_w, 25), cv2.FONT_HERSHEY_SIMPLEX,
               0.55, (200,200,200), 1)
    cv2.putText(frame, f"{current_scene}",
               (w-box_w, 52), cv2.FONT_HERSHEY_SIMPLEX,
               0.8, (0,255,255), 2)
    cv2.putText(frame, f"{scene_confidence:.1f}%",
               (w-box_w, 70), cv2.FONT_HERSHEY_SIMPLEX,
               0.5, (180,180,180), 1)

    # ── BOTTOM — Sentence in progress ────────────────────────
    # Shows unique words detected so far
    # Clears after SENTENCE_PAUSE_SEC of stillness
    if sentence_words:
        sentence_text = "  →  ".join(sentence_words)

        # Background
        cv2.rectangle(frame, (0, h-70), (w, h-35), (0,0,40), -1)
        cv2.rectangle(frame, (0, h-70), (w, h-35), (50,50,150), 2)

        cv2.putText(frame, "Sentence:",
                   (10, h-52), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (150,150,255), 1)

        cv2.putText(frame, sentence_text,
                   (10, h-38), cv2.FONT_HERSHEY_SIMPLEX,
                   0.75, (255,255,255), 2)

        # Show pause timer
        remaining = max(0, SENTENCE_PAUSE_SEC - stillness)
        if stillness > 0.2:
            cv2.putText(frame, f"clears in {remaining:.1f}s",
                       (w-160, h-52), cv2.FONT_HERSHEY_SIMPLEX,
                       0.45, (150,150,150), 1)

    # ── BOTTOM status bar ─────────────────────────────────────
    hand_color = (0,255,0) if hand_detected else (0,0,255)
    cv2.putText(frame,
               f"Hand:{'ON' if hand_detected else 'OFF'} | "
               f"Buf:{len(buffer)}/{MAX_FRAMES} | "
               f"Words:{len(sentence_words)}",
               (10, h-10),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5, hand_color, 1)

    # ── Sliding window progress bar ───────────────────────────
    bar_x, bar_y = 10, h-28
    bar_w, bar_h = 220, 10
    fill_ratio   = min(len(buffer) / MAX_FRAMES, 1.0)
    fill_w       = int(bar_w * fill_ratio)

    if len(buffer) < MIN_FRAMES:
        bar_color = (0, 0, 200)
    elif confidence / 100.0 >= get_dynamic_threshold(len(buffer)):
        bar_color = (0, 220, 0)
    else:
        bar_color = (0, 200, 220)

    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x+bar_w, bar_y+bar_h), (60,60,60), -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x+fill_w, bar_y+bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y),
                 (bar_x+bar_w, bar_y+bar_h), (150,150,150), 1)

    cv2.imshow("NSL Sign Recognition — Q:quit  SPACE:reset", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if sentence_words:
            log_sentence(sentence_words)
        break
    elif key == ord(' '):
        buffer.clear()
        pred_history.clear()
        sentence_words  = []
        last_committed  = None
        predicted       = "Show a sign..."
        confidence      = 0.0
        top3_preds      = []
        sentence_finalized = False
        print("🔄 Reset!")

# ── Cleanup ───────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()
pose_landmarker.close()
print("✅ Done")