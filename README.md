# Real-Time Age & Gender Detection
### Custom CNN (Keras) + Haar Cascade Face Detection — Trained on UTKFace

Detects faces from a webcam (or a single image) and predicts **age** and **gender** in real time using a custom-trained multi-output CNN. Fully local — no API calls, no internet required at inference time.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│            Webcam / Image Input              │
└──────────────────────┬───────────────────────┘
                        │ BGR frame
┌──────────────────────▼───────────────────────┐
│          Haar Cascade Face Detector           │
│   haarcascade/haarcascade_frontalface_default │
│              .xml                             │
└──────────────────────┬───────────────────────┘
                        │ face ROI (grayscale, NxM)
┌──────────────────────▼───────────────────────┐
│         Custom Multi-Output CNN (Keras)       │
│  Input: 128×128×1 grayscale                   │
│  Rescaling(1/255) ← baked into the model      │
│  4× [Conv2D → MaxPool]  (32→64→128→256)       │
│  Flatten (shared trunk)                       │
│  ├─ Dense(256) → Dropout → output_gender      │
│  └─ Dense(256) → Dropout → output_age         │
└──────────────────────┬───────────────────────┘
                        │ gender (sigmoid), age (linear)
┌──────────────────────▼───────────────────────┐
│        OpenCV Overlay (box + label)           │
└────────────────────────────────────────────────┘
```

## Training Pipeline

```
UTKFace dataset
  filenames: {age}_{gender}_{race}_{date}.jpg
    │
    ▼
Parse filename → age, gender labels
    │
    ▼
train / val / test split (80 / 10 / 10)
    │
    ▼
tf.data pipeline: decode jpeg → resize 128×128 → grayscale
    │
    ▼
Data Augmentation (flip, rotation, zoom)   [train only]
    │
    ▼
Rescaling(1/255)                            [inside the model]
    │
    ▼
Custom CNN — multi-output (gender + age)
    │
    ▼
models/age_gender_custom_cnn_v1.keras
```

## Model Details

| Output | Type | Activation | Loss |
|--------|------|------------|------|
| `output_gender` | Binary classification (0 = Male, 1 = Female) | sigmoid | binary_crossentropy |
| `output_age` | Regression (years) | linear | mae |

Input is always **128×128, single-channel grayscale**, raw pixel values in `[0, 255]`. Normalization happens *inside* the model via a `Rescaling(1./255.)` layer — see [Known Issues](#known-issues--lessons-learned) below for why this matters.

## Setup (First Time)

```bash
git clone <your-repo-url>
cd age-gender-detection

# Inference-only setup
bash setup.sh

# If you also want to retrain the model
bash setup.sh --train
```

## Running

```bash
source .venv/bin/activate

# Real-time webcam detection
python realtime_detection.py

# Single image (no webcam needed)
python scripts/predict_image.py --image path/to/photo.jpg --save output.jpg
```

Press `q` to quit the webcam window.

## Retraining the Model

The model was trained on [UTKFace](https://susanqq.github.io/UTKFace/). Download it, then either:

**Option A — Notebook** (includes EDA + plots):
```bash
jupyter notebook notebooks/ageandgender.ipynb
```

**Option B — Script:**
```bash
python scripts/train.py --data-path /path/to/UTKFace --epochs 50
```

The final model is saved to `models/age_gender_custom_cnn_v1.keras`.

## Project Structure

```
age-gender-detection/
├── realtime_detection.py     # Live webcam inference
├── scripts/
│   ├── train.py               # Script version of the training notebook
│   └── predict_image.py       # Single-image inference (no webcam)
├── notebooks/
│   └── ageandgender.ipynb     # Full training notebook (EDA, training, eval plots)
├── models/
│   └── age_gender_custom_cnn_v1.keras
├── haarcascade/
│   └── haarcascade_frontalface_default.xml
├── requirements.txt           # Inference-only deps
├── requirements-train.txt     # Extra deps for (re)training
├── setup.sh
└── LICENSE
```

## Known Issues / Lessons Learned

**Double-normalization bug (fixed):** the model's first real layer is `Rescaling(1./255.)`, meaning it expects raw `0–255` pixel values as input and divides by 255 itself. An earlier version of `realtime_detection.py` *also* divided the face crop by 255 before feeding it to the model, so every input was effectively scaled down to `~0–0.004` — close enough to a blank frame that the network just output its learned average for every face (age stuck around 45–48, gender always low-confidence "Male", regardless of who or what was in front of the camera). The fix was simply to stop normalizing manually and let the model's own `Rescaling` layer do it, since that's exactly how `scripts/train.py` / the notebook feed images during training.

**Dataset skew:** UTKFace is skewed toward adult faces; expect lower accuracy on children and the elderly unless you augment with a more balanced dataset.

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
→ Activate the venv first: `source .venv/bin/activate`

**Webcam doesn't open / black window**
→ Try a different camera index: `cv2.VideoCapture(1)` instead of `0` in `realtime_detection.py`.

**Predictions look constant / barely change across faces**
→ Check you're not normalizing pixel values before passing them to the model — see [Known Issues](#known-issues--lessons-learned).

**`Failed to load Haar Cascade`**
→ Make sure you cloned the repo with the `haarcascade/` folder intact; the path is resolved relative to the script, not your current working directory.

## License

MIT — see [LICENSE](LICENSE).
