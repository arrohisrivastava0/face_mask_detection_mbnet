"""
train_model.py — Fine-tune MobileNetV2 on the face-mask dataset
Dataset: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
         (two folders: with_mask / without_mask)

Usage:
    python train_model.py --dataset dataset
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (AveragePooling2D, Flatten, Dense, Dropout, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
INIT_LR   = 1e-4
EPOCHS    = 20
BS        = 32
IMG_SIZE  = 224

def main(dataset_path, output_weights):
    print("[INFO] Loading images …")
    data, labels = [], []

    for category in os.listdir(dataset_path):
        cat_path = os.path.join(dataset_path, category)
        if not os.path.isdir(cat_path):
            continue
        for img_file in os.listdir(cat_path):
            img_path = os.path.join(cat_path, img_file)
            try:
                img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                img = img_to_array(img)
                img = preprocess_input(img)
                data.append(img)
                labels.append(category)
            except Exception:
                pass

    data   = np.array(data,  dtype="float32")
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    print(f"[INFO] Classes detected: {lb.classes_}")
    print(f"[INFO] Total images:     {len(data)}")

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42)

    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    # ── Build model ───────────────────────────────────────────────────────────
    base = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    for layer in base.layers:
        layer.trainable = False

    head = base.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)
    model = Model(inputs=base.input, outputs=head)

    print("[INFO] Compiling model …")
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
        metrics=["accuracy"],
    )

    print("[INFO] Training head …")
    H = model.fit(
        aug.flow(X_train, y_train, batch_size=BS),
        steps_per_epoch=len(X_train) // BS,
        validation_data=(X_test, y_test),
        validation_steps=len(X_test) // BS,
        epochs=EPOCHS,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("[INFO] Evaluating …")
    preds = model.predict(X_test, batch_size=BS)
    preds = np.argmax(preds, axis=1)
    print(classification_report(
        np.argmax(y_test, axis=1), preds,
        target_names=lb.classes_))

    # ── Save weights ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    model.save_weights(output_weights)
    print(f"[INFO] Weights saved → {output_weights}")

    # ── Plot training curve ───────────────────────────────────────────────────
    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(H.history["loss"],      label="train_loss")
    axes[0].plot(H.history["val_loss"],  label="val_loss")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(H.history["accuracy"],     label="train_acc")
    axes[1].plot(H.history["val_accuracy"], label="val_acc")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    print("[INFO] Training plot saved → training_plot.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
        help="Path to dataset folder (with_mask / without_mask)")
    ap.add_argument("--weights", default="model/mask_detector.h5",
        help="Output path for saved weights")
    args = ap.parse_args()
    main(args.dataset, args.weights)
