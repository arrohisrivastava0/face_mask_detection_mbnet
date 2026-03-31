"""
Face Mask Detection - Real-time Webcam
Uses OpenCV face detector + a simple MobileNetV2-based classifier
"""

import cv2
import numpy as np
import argparse
import sys
import os

# ── Suppress TF/Keras noise ──────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    AveragePooling2D, Flatten, Dense, Dropout, Input
)


# ── Build / load model ───────────────────────────────────────────────────────
def build_model(weights_path=None):
    """
    Constructs MobileNetV2-based binary classifier.
    If a .h5 weights file is provided it is loaded; otherwise imagenet
    weights are used as-is (demo mode — accuracy will vary).
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3)),
    )
    head = base.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)

    model = Model(inputs=base.input, outputs=head)

    if weights_path and os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print(f"[INFO] Loaded fine-tuned weights from '{weights_path}'")
    else:
        print("[INFO] No fine-tuned weights found — running in demo mode.")
        print("       See README.md for instructions to download/train weights.")

    return model


# ── Load OpenCV face detector ────────────────────────────────────────────────
def load_face_detector():
    proto = cv2.data.haarcascades  # fallback path
    # Prefer DNN-based detector (more accurate)
    prototxt = os.path.join("model", "deploy.prototxt")
    caffemodel = os.path.join("model", "res10_300x300_ssd_iter_140000.caffemodel")

    if os.path.isfile(prototxt) and os.path.isfile(caffemodel):
        print("[INFO] Using DNN face detector.")
        return "dnn", cv2.dnn.readNet(prototxt, caffemodel)
    else:
        print("[INFO] DNN model files not found — using Haar Cascade detector.")
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        return "haar", cv2.CascadeClassifier(cascade_path)


# ── Detect faces ─────────────────────────────────────────────────────────────
def detect_faces(frame, detector_type, detector):
    h, w = frame.shape[:2]
    faces = []

    if detector_type == "dnn":
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        detector.setInput(blob)
        detections = detector.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                faces.append((x1, y1, x2, y2))
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        for (x, y, fw, fh) in detected:
            faces.append((x, y, x + fw, y + fh))

    return faces


# ── Predict mask for one face ROI ────────────────────────────────────────────
def predict_mask(face_roi, model):
    face = cv2.resize(face_roi, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    preds = model.predict(face, verbose=0)[0]
    # index 0 = mask, index 1 = no_mask  (matches training label order)
    mask_prob, no_mask_prob = preds[0], preds[1]
    label = "Mask" if mask_prob > no_mask_prob else "No Mask"
    confidence = max(mask_prob, no_mask_prob)
    return label, confidence


# ── Main loop ────────────────────────────────────────────────────────────────
def run(camera_index=0, weights_path=None):
    print("[INFO] Loading model …")
    model = build_model(weights_path)

    print("[INFO] Loading face detector …")
    detector_type, detector = load_face_detector()

    print("[INFO] Starting webcam …")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {camera_index}.")
        sys.exit(1)

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror effect
        faces = detect_faces(frame, detector_type, detector)

        for (x1, y1, x2, y2) in faces:
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            label, confidence = predict_mask(roi, model)

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            text = f"{label}: {confidence * 100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)
            cv2.putText(frame, text, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Stats overlay
        cv2.putText(frame, "Face Mask Detector | press 'q' to quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Face Mask Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Stream closed.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Face Mask Detector")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--weights", type=str, default="model/mask_detector.h5",
                        help="Path to fine-tuned .h5 weights (optional)")
    args = parser.parse_args()

    run(camera_index=args.camera, weights_path=args.weights)
