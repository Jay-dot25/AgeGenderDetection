"""
Run age/gender prediction on a single image file (no webcam needed).

Usage:
    python scripts/predict_image.py --image path/to/photo.jpg
"""

import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender_custom_cnn_v1.keras")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade", "haarcascade_frontalface_default.xml")


def parse_args():
    parser = argparse.ArgumentParser(description="Age/gender prediction on a single image")
    parser.add_argument("--image", required=True, help="Path to an input image")
    parser.add_argument("--save", help="Optional path to save the annotated output image")
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from {CASCADE_PATH}")

    frame = cv2.imread(args.image)
    if frame is None:
        raise IOError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

    if len(faces) == 0:
        print("No faces detected.")
        return

    for (x, y, w, h) in faces:
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (128, 128))
        face = face.astype("float32")  # model has its own Rescaling(1/255) layer built in
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        gender_pred, age_pred = model.predict(face, verbose=0)
        gender_score = float(gender_pred[0][0])
        age_value = float(age_pred[0][0])

        gender = "Female" if gender_score > 0.5 else "Male"
        age = max(0, int(age_value))

        label = f"{gender}, {age} ({gender_score:.2f})"
        print(label)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"Annotated image saved to: {args.save}")
    else:
        cv2.imshow("Age & Gender Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
