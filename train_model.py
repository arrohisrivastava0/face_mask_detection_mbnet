"""
train_model.py — Fine-tune MobileNetV2 on the face-mask dataset
Memory-efficient version using flow_from_directory (no full dataset in RAM)

Usage:
    python train_model.py --dataset dataset
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
INIT_LR  = 1e-4
EPOCHS   = 10        # reduced for speed; increase to 20 for better accuracy
BS       = 16        # smaller batch = less RAM
IMG_SIZE = 224


def main(dataset_path, output_weights):
    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    # ── Data generators (images loaded batch-by-batch, not all at once) ───────
    print("[INFO] Setting up data generators ...")

    train_aug = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    train_gen = train_aug.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BS,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = train_aug.flow_from_directory(
        dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BS,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    print(f"[INFO] Classes: {train_gen.class_indices}")
    print(f"[INFO] Training samples:   {train_gen.samples}")
    print(f"[INFO] Validation samples: {val_gen.samples}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("[INFO] Building model ...")
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    )
    for layer in base.layers:
        layer.trainable = False

    head = base.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2, activation="softmax")(head)
    model = Model(inputs=base.input, outputs=head)

    print("[INFO] Compiling ...")
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=INIT_LR),
        metrics=["accuracy"],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("[INFO] Training ... (this may take 10-20 mins on CPU)")
    H = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BS,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BS,
        epochs=EPOCHS,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("[INFO] Evaluating ...")
    val_gen.reset()
    preds = model.predict(val_gen, steps=val_gen.samples // BS + 1, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = val_gen.classes[:len(pred_labels)]
    class_names  = list(val_gen.class_indices.keys())
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save_weights(output_weights)
    print(f"[INFO] Weights saved -> {output_weights}")

    # ── Plot ──────────────────────────────────────────────────────────────────
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
    print("[INFO] Plot saved -> training_plot.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Path to dataset folder (with_mask / without_mask subfolders)")
    ap.add_argument("--weights", default="model/mask_detector.weights.h5",
                    help="Output path for saved weights")
    args = ap.parse_args()
    main(args.dataset, args.weights)
