"""
Real-time age & gender detection from a webcam.

Haar Cascade finds faces, the custom multi-output CNN predicts age and gender,
and detection_utils merges duplicate boxes and smooths the numbers over time.

Usage:
    python realtime_detection.py                 # default camera, quiet
    python realtime_detection.py --camera 1      # a different webcam
    python realtime_detection.py --debug         # raw + smoothed readout
    python realtime_detection.py --no-smooth     # per-frame values, as they come
    python realtime_detection.py --no-nms        # show every raw Haar box

Press 'q' in the video window to quit.
"""

import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from detection_utils import FaceTracker, format_label, non_max_suppression

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender_custom_cnn_v1.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")

IMG_SIZE = 128           # the model input is 128x128x1 grayscale
WINDOW_NAME = "Age & Gender Detection"


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time age/gender detection on a webcam")
    parser.add_argument(
        "--camera", type=int, default=0,
        help="cv2.VideoCapture device index (use 1 if 0 is taken or returns black frames)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="print raw and smoothed gender score / age for every detected face on every frame",
    )
    parser.add_argument("--min-face", type=int, default=60, help="minimum face size, in pixels")
    parser.add_argument("--scale-factor", type=float, default=1.3, help="Haar detector scale factor")
    parser.add_argument(
        "--smooth", type=int, default=5,
        help="EMA window per face, in frames. 1 = no smoothing (default: 5)",
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="shorthand for --smooth 1: keep tracking and label hysteresis, drop the averaging",
    )
    parser.add_argument(
        "--gender-margin", type=float, default=0.05,
        help="hysteresis half-band around 0.5; a label only flips past 0.5+/-margin. "
             "0.0 reproduces a plain 0.5 threshold",
    )
    parser.add_argument(
        "--nms-iou", type=float, default=0.3,
        help="merge boxes above this IoU into one detection (default: 0.3)",
    )
    parser.add_argument("--no-nms", action="store_true", help="disable duplicate-box merging")
    return parser.parse_args()


def load_model_and_cascade():
    """Loads the CNN and the face detector. Paths resolve next to this file, not the cwd."""
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(
            f"Model file not found: {MODEL_PATH}\n"
            "Clone the repo with the models/ folder intact, or retrain with scripts/train.py."
        )

    model = load_model(MODEL_PATH)
    print("Model loaded successfully")

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {CASCADE_PATH}")

    return model, face_cascade


def _feed(model, batch):
    """
    Wrap the batch as {input_name: batch} when the model has one named input.

    Keras 3 emits "UserWarning: The structure of `inputs` doesn't match the
    expected structure" on every call that passes a bare array to a Functional
    model whose input layer is named -- including model.predict(), so this was
    noisy before too. During a webcam session it prints twice per new face
    count, burying real output. Feeding by name is bit-identical (verified
    delta 0.0 on gender and age) and silent.
    """
    try:
        if len(model.inputs) == 1:
            name = model.inputs[0].name.split(":")[0]
            if name:
                return {name: batch}
    except Exception:
        pass
    return batch


def predict_faces(model, gray, faces):
    """
    Runs one batched forward pass for every face in the frame.

    Returns (gender_scores, ages) as 1-D float arrays aligned with `faces`.
    """
    crops = np.stack(
        [cv2.resize(gray[y : y + h, x : x + w], (IMG_SIZE, IMG_SIZE)).astype("float32") for (x, y, w, h) in faces]
    )
    crops = np.expand_dims(crops, axis=-1)  # (N, 128, 128, 1)

    # Feed RAW 0-255 pixels. The model's own Rescaling(1/255) layer normalizes;
    # dividing here too collapses every prediction to the dataset mean.

    try:
        # Direct call is much cheaper per frame than predict(), which rebuilds
        # its data pipeline machinery on every invocation.
        gender_pred, age_pred = model(_feed(model, crops), training=False)
        gender_pred, age_pred = gender_pred.numpy(), age_pred.numpy()
    except Exception:
        gender_pred, age_pred = model.predict(_feed(model, crops), verbose=0)

    return np.asarray(gender_pred).reshape(-1), np.asarray(age_pred).reshape(-1)


def analyze_frame(model, face_cascade, tracker, frame, args):
    """
    Detect -> merge -> predict -> smooth, for one frame.

    Drawn in place on `frame`; returns the list of row dicts it drew so a
    --debug caller (or a test) can inspect the numbers without a GUI.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = face_cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=5,
        minSize=(args.min_face, args.min_face),
    )
    faces = found if args.no_nms else non_max_suppression(found, iou_threshold=args.nms_iou)
    if len(faces) == 0:
        return []

    gender_scores, ages = predict_faces(model, gray, faces)
    rows = tracker.update(faces, gender_scores, ages)

    for row in rows:
        x, y, w, h = row["box"]
        label = format_label(row)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Console I/O is synchronous on Windows and will stall the loop, so
        # this is opt-in rather than per-face-per-frame by default.
        if args.debug:
            print(
                f"[face {row['id']}] gender {row['raw_gender_score']:.4f} -> {row['gender_score']:.4f}"
                f" | age {row['raw_age']:.2f} -> {row['age']:.2f} | {label}",
                flush=True,
            )

    return rows


def main():
    args = parse_args()
    if args.no_smooth:
        args.smooth = 1

    model, face_cascade = load_model_and_cascade()
    tracker = FaceTracker(window=args.smooth, gender_margin=args.gender_margin)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera index {args.camera}. "
            "Try --camera 1, and close anything holding the webcam (Windows Camera app, Teams, Zoom)."
        )

    try:
        # Fail here with a clear message rather than 60 lines into the loop.
        try:
            cv2.namedWindow(WINDOW_NAME)
        except cv2.error:
            raise SystemExit(
                "This OpenCV build has no GUI support (headless). "
                "Use scripts/predict_image.py --save instead, or install libgl1."
            )

        print(
            f"Smoothing window: {tracker.window} frame(s) | gender hysteresis: +/-{tracker.gender_margin:.2f}"
            f" | duplicate-box merging: {'off' if args.no_nms else f'IoU {args.nms_iou:.2f}'}"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stopped returning frames; exiting.")
                break

            analyze_frame(model, face_cascade, tracker, frame, args)

            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always runs -- on Ctrl+C, on an exception, on 'q'. Skipping this leaves
        # the capture device locked on Windows, which looks like a broken camera
        # on the next run.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
