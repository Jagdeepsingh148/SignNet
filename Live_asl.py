import cv2
import torch
import collections
import os
import time
import mediapipe as mp
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModelForImageClassification

# Access new Tasks API via mp.tasks
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
BaseOptions = mp.tasks.BaseOptions

# best hand-crop model, produced by: python trainV2.py --input crop --robust
MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "effb0_crop_robust", "model")
CAMERA_INDEX = 0
SMOOTH_WINDOW = 7
MIN_CONF = 0.60
PAD = 0.25  # extra margin around the detected hand

# Path to the downloaded hand_landmarker.task model (same folder as this script)
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

device = "cuda" if torch.cuda.is_available() else "cpu"
# identical to the eval transform used in trainV2.py
preprocess = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device).eval()

# Set up MediaPipe HandLandmarker (new Tasks API)
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    raise RuntimeError("Could not open camera. Check CAMERA_INDEX or camera permissions.")

history = collections.deque(maxlen=SMOOTH_WINDOW)
frame_ts = 0  # monotonic timestamp in ms for VIDEO mode
fps, t_prev = 0.0, time.perf_counter()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to mediapipe Image and detect
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    frame_ts += 33  # ~30 fps increment
    result = landmarker.detect_for_video(mp_image, frame_ts)

    label, conf = "-", 0.0
    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]  # first hand

        # Draw landmarks on frame
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Draw connections
        for conn in HandLandmarksConnections.HAND_CONNECTIONS:
            start_lm = landmarks[conn.start]
            end_lm = landmarks[conn.end]
            pt1 = (int(start_lm.x * w), int(start_lm.y * h))
            pt2 = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Compute bounding box from landmarks
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        cx_box, cy_box = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * PAD)
        x1, y1 = int(max(cx_box - side / 2, 0)), int(max(cy_box - side / 2, 0))
        x2, y2 = int(min(cx_box + side / 2, w)), int(min(cy_box + side / 2, h))

        roi = rgb[y1:y2, x1:x2]
        if roi.size > 0:
            x = preprocess(Image.fromarray(roi)).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(pixel_values=x).logits, dim=-1)[0]
            idx = probs.argmax().item()
            conf = probs[idx].item()
            history.append(model.config.id2label[idx] if conf >= MIN_CONF else None)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        history.append(None)

    votes = [l for l in history if l is not None]
    if votes:
        label = max(set(votes), key=votes.count)

    cv2.putText(frame, f"{label}  {conf:.2f}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    t_now = time.perf_counter()
    fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
    t_prev = t_now
    cv2.putText(frame, f"{fps:.1f} FPS", (w - 140, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Live ASL (hand tracking)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
