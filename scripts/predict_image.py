"""
Run age/gender prediction on a single image file (no webcam needed).

This is also the headless path for servers and containers: pass --save and
nothing ever tries to open a GUI window.

Usage:
    python scripts/predict_image.py --image path/to/photo.jpg
    python scripts/predict_image.py --image path/to/photo.jpg --save out/annotated.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender_custom_cnn_v1.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")

IMG_SIZE = 128           # the model input is 128x128x1 grayscale
GENDER_THRESHOLD = 0.5   # UTKFace labels: 0 = Male, 1 = Female


def parse_args():
    parser = argparse.ArgumentParser(description="Age/gender prediction on a single image")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--save", help="Optional path to save the annotated output image")
    parser.add_argument("--min-face", type=int, default=60, help="minimum face size, in pixels")
    parser.add_argument("--scale-factor", type=float, default=1.3, help="Haar detector scale factor")
    return parser.parse_args()


def predict_faces(model, gray, faces):
    """
    One batched forward pass for every face found. Returns (gender_scores, ages)
    aligned with `faces`.
    """
    crops = np.stack(
        [cv2.resize(gray[y : y + h, x : x + w], (IMG_SIZE, IMG_SIZE)).astype("float32") for (x, y, w, h) in faces]
    )
    crops = np.expand_dims(crops, axis=-1)  # (N, 128, 128, 1)

    # Feed RAW 0-255 pixels -- the model has its own Rescaling(1/255) layer built in.
    gender_pred, age_pred = model.predict(crops, verbose=0)

    return np.asarray(gender_pred).reshape(-1), np.asarray(age_pred).reshape(-1)


def save_annotated(frame, path):
    """
    Writes the image and fails loudly if OpenCV could not actually write it.

    imwrite has two distinct failure modes, both of which used to go unnoticed:
    it returns False for a missing parent directory, and it *raises* cv2.error
    for an extension it has no encoder for.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)  # imwrite does NOT create missing folders

    try:
        wrote = cv2.imwrite(path, frame)
    except cv2.error as e:
        raise SystemExit(f"cv2.imwrite could not write {path}: {e}")

    if not wrote:
        raise SystemExit(
            f"cv2.imwrite returned False for {path}. OpenCV only writes extensions it "
            "knows (.png, .jpg, .jpeg, .bmp, .tiff) and cannot write to a locked folder."
        )

    print(f"Annotated image saved to: {os.path.abspath(path)}")


def main():
    args = parse_args()

    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Model file not found: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {CASCADE_PATH}")

    frame = cv2.imread(args.image)
    if frame is None:
        raise IOError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=5,
        minSize=(args.min_face, args.min_face),
    )

    if len(faces) == 0:
        print("No faces detected.")
        if args.save:
            # Still emit the file so batch scripts get a consistent output set.
            save_annotated(frame, args.save)
        return 1

    gender_scores, age_values = predict_faces(model, gray, faces)

    for i, (x, y, w, h) in enumerate(faces):
        gender_score = float(gender_scores[i])
        age_value = float(age_values[i])

        gender = "Female" if gender_score > GENDER_THRESHOLD else "Male"
        age = max(0, int(age_value))

        label = f"{gender}, {age} ({gender_score:.2f})"
        print(label)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if args.save:
        save_annotated(frame, args.save)
    else:
        cv2.imshow("Age & Gender Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
