# AI Face Recognition

A simple, beginner-friendly face recognition project built with **OpenCV**
(Haar Cascade for detection + LBPH for recognition). No GPU or deep learning
framework required.

## Project Structure

```
AI_Face_Recognition/
│
├── dataset/            # Captured face images, organized by person
├── model/               # Trained model + label mappings (generated)
├── capture_images.py    # Step 1: capture face images from webcam
├── train_model.py       # Step 2: train the recognizer on captured images
├── predict_image.py     # Step 3a: recognize a face in a static image
├── webcam_predict.py     # Step 3b: recognize faces live from webcam
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Capture training images
Run this once per person you want the model to recognize:
```bash
python capture_images.py
```
You'll be prompted for a name (e.g. `alice`). It opens your webcam and
saves ~60 cropped face images to `dataset/alice/`.

Repeat for each person (`bob`, `carol`, etc.) so `dataset/` ends up like:
```
dataset/
├── alice/
│   ├── alice_1.jpg
│   └── ...
└── bob/
    ├── bob_1.jpg
    └── ...
```

### 2. Train the model
```bash
python train_model.py
```
This reads every image in `dataset/`, trains an LBPH face recognizer, and
saves:
- `model/trained_model.yml` – the trained recognizer
- `model/labels.pickle` – maps numeric IDs back to person names

### 3. Recognize faces

**On a static image:**
```bash
python predict_image.py path/to/photo.jpg
```
Draws a box + name around any recognized face and saves
`prediction_output.jpg`.

**Live via webcam:**
```bash
python webcam_predict.py
```
Press `q` to quit.

## Notes & Tuning

- **Confidence threshold**: Both prediction scripts use
  `CONFIDENCE_THRESHOLD = 70`. LBPH confidence is a *distance* — lower
  means a better match. If you get too many "Unknown" results, raise the
  threshold; if you get false positives, lower it.
- **More/better images = better accuracy.** Capture faces at different
  angles, expressions, and lighting conditions.
- **Camera index**: If `cv2.VideoCapture(0)` doesn't open your camera, try
  `1`, `2`, etc.
- This is a lightweight classical CV approach (not a deep-learning
  embedding model like FaceNet/ArcFace), so accuracy is good for small,
  controlled datasets (a handful of known people) but won't scale to
  large-scale face recognition.

## Requirements

See `requirements.txt`. Core dependency is `opencv-contrib-python`, which
includes the `cv2.face` module needed for LBPH recognition.
