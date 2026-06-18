import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("age_gender_custom_cnn_v1.keras")
print("Model loaded successfully")

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Preprocess exactly like training
        face = cv2.resize(face, (128, 128))
        face = face.astype("float32")  # model has its own Rescaling(1/255) layer built in

        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        try:
            gender_pred, age_pred = model.predict(
                face,
                verbose=0
            )

            gender_score = float(gender_pred[0][0])
            age_value = float(age_pred[0][0])

            # -------------------------
            # Gender Mapping
            # -------------------------
            #
            # If labels were:
            # 0 = Male, 1 = Female
            #
            gender = "Female" if gender_score > 0.5 else "Male"

            # If output is reversed, change to:
            # gender = "Male" if gender_score > 0.5 else "Female"
            #
            # -------------------------

            age = max(0, int(age_value))

            label = (
                f"{gender}, {age} "
                f"({gender_score:.2f})"
            )

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Debug output
            print(
                f"Gender Score: {gender_score:.4f} | "
                f"Age: {age_value:.2f}"
            )

        except Exception as e:
            print("Prediction Error:", e)

    cv2.imshow(
        "Age & Gender Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()