# 😷 Face Mask Detection — Real-Time Webcam

A real-time face mask detection system using **MobileNetV2** (transfer learning) and **OpenCV**.  
Detects whether a person in front of the webcam is wearing a face mask or not, with live bounding boxes and confidence scores.

---

## 📁 Project Structure

```
face-mask-detection/
├── detect_mask.py       # Main script — real-time webcam detection
├── train_model.py       # Optional — fine-tune model on custom dataset
├── requirements.txt     # Python dependencies
├── model/               # Place your weights file here (see below)
│   └── mask_detector.h5 # (download or train — see instructions)
└── dataset/             # Optional — for training
    ├── with_mask/
    └── without_mask/
```

---

## ⚙️ Environment Setup

### 1. Prerequisites
- Python **3.8 – 3.10** (recommended)
- pip
- A working webcam

### 2. Clone the Repository
```bash
git clone https://github.com/{your-username}/face-mask-detection.git
cd face-mask-detection
```

### 3. Create a Virtual Environment (recommended)
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Real-Time Webcam Detection
```bash
python detect_mask.py
```

| Argument | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index (use `1`, `2` … for external cameras) |
| `--weights` | `model/mask_detector.h5` | Path to fine-tuned weights (optional) |

Press **`q`** to quit the webcam window.

> **Note:** If no weights file is found, the system runs in **demo mode** using raw ImageNet weights. For accurate predictions, download or train the weights (see below).

---

## 🧠 Model Weights

### Option A — Download Pre-trained Weights
Download `mask_detector.h5` from the dataset repository below and place it inside the `model/` folder:

```
model/mask_detector.h5
```

### Option B — Train From Scratch

**Step 1:** Download the dataset:
```
https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset
```
Place it as:
```
dataset/
  with_mask/
  without_mask/
```

**Step 2:** Run training:
```bash
python train_model.py --dataset dataset
```
Trained weights are saved automatically to `model/mask_detector.h5`.

---

## 🔍 How It Works

1. **Face Detection** — OpenCV's DNN (SSD ResNet10) or Haar Cascade detects faces in each webcam frame.
2. **Preprocessing** — Each detected face is resized to 224×224 and preprocessed for MobileNetV2.
3. **Classification** — The MobileNetV2-based classifier predicts `Mask` or `No Mask` with a confidence score.
4. **Visualization** — Results are drawn live on the webcam feed with colored bounding boxes (🟢 Mask / 🔴 No Mask).

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | Deep learning framework |
| `opencv-python` | Webcam access & image processing |
| `numpy` | Numerical operations |
| `scikit-learn` | Data splitting & evaluation metrics |
| `matplotlib` | Training curve visualization |

---

## 📌 Troubleshooting

| Issue | Fix |
|---|---|
| `Cannot open camera index 0` | Try `--camera 1` or check webcam connection |
| Slow performance | Reduce frame resolution or use a GPU |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| TensorFlow install issues | Use Python 3.8–3.10; avoid 3.12 |

---

## 📄 License
MIT License — free to use and modify.
