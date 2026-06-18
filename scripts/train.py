"""
Train the multi-output (gender + age) CNN on UTKFace.

This is a cleaned-up, script form of notebooks/ageandgender.ipynb.
Use the notebook if you want the inline plots / EDA; use this if you
just want to (re)train the model from the command line.

Usage:
    python scripts/train.py --data-path /path/to/UTKFace --epochs 50
"""

import argparse
import glob
import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    Rescaling,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def parse_args():
    parser = argparse.ArgumentParser(description="Train age/gender CNN on UTKFace")
    parser.add_argument("--data-path", required=True, help="Path to the UTKFace image folder")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "age_gender_custom_cnn_v1.keras"),
        help="Where to save the final .keras model",
    )
    return parser.parse_args()


def parse_utk_filename(filepath):
    """UTKFace filenames look like: {age}_{gender}_{race}_{date}.jpg"""
    try:
        basename = os.path.basename(filepath)
        parts = basename.split("_")
        age = int(parts[0])
        gender = int(parts[1])
        return filepath, age, gender
    except Exception:
        return None


def load_dataframe(data_path):
    image_files = glob.glob(os.path.join(data_path, "*.jpg"))
    print(f"Found {len(image_files)} image files.")

    data = [parse_utk_filename(f) for f in image_files]
    data = [d for d in data if d is not None]

    df = pd.DataFrame(data, columns=["filepath", "age", "gender"])
    df = df[(df["age"] > 0) & (df["age"] <= 100)]

    print(f"Loaded and parsed {len(df)} valid images.")
    return df


def make_parse_fn(img_size):
    def parse_image_and_labels(filepath, age, gender):
        """Loads image, resizes, converts to grayscale, and prepares labels."""
        img = tf.io.read_file(filepath)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.image.rgb_to_grayscale(img)

        age = tf.cast(age, dtype=tf.float32)
        gender = tf.cast(gender, dtype=tf.float32)

        return img, {"output_gender": gender, "output_age": age}

    return parse_image_and_labels


def create_dataset(df, parse_fn, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (df["filepath"].values, df["age"].values, df["gender"].values)
    )
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.shuffle(buffer_size=len(df))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def build_custom_cnn_model(input_shape):
    """
    Custom multi-output CNN: shared conv trunk, two heads
    (gender classification + age regression).

    NOTE: Rescaling(1./255.) lives INSIDE the model. Any inference
    script should feed raw 0-255 pixel values, NOT pre-normalized
    ones -- normalizing twice silently breaks predictions.
    """
    inputs = Input(shape=input_shape, name="input_layer")

    data_augmentation = keras.Sequential(
        [RandomFlip("horizontal"), RandomRotation(0.1), RandomZoom(0.1)],
        name="data_augmentation",
    )

    x = data_augmentation(inputs)
    x = Rescaling(1.0 / 255.0)(x)

    x = Conv2D(32, (3, 3), activation="relu", padding="valid", name="conv2d")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d")(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="valid", name="conv2d_1")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_1")(x)

    x = Conv2D(128, (3, 3), activation="relu", padding="valid", name="conv2d_2")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_2")(x)

    x = Conv2D(256, (3, 3), activation="relu", padding="valid", name="conv2d_3")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_3")(x)

    x_shared = Flatten(name="flatten")(x)

    gender_branch = Dense(256, activation="relu", name="dense")(x_shared)
    gender_branch = Dropout(0.5, name="dropout")(gender_branch)
    output_gender = Dense(1, activation="sigmoid", name="output_gender")(gender_branch)

    age_branch = Dense(256, activation="relu", name="dense_1")(x_shared)
    age_branch = Dropout(0.5, name="dropout_1")(age_branch)
    output_age = Dense(1, activation="linear", name="output_age")(age_branch)

    return Model(inputs=inputs, outputs=[output_gender, output_age], name="Custom_CNN_Age_Gender")


def main():
    args = parse_args()

    df = load_dataframe(args.data_path)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples:       {len(test_df)}")

    parse_fn = make_parse_fn(args.img_size)
    train_ds = create_dataset(train_df, parse_fn, args.batch_size, augment=True)
    val_ds = create_dataset(val_df, parse_fn, args.batch_size)
    test_ds = create_dataset(test_df, parse_fn, args.batch_size)

    model = build_custom_cnn_model((args.img_size, args.img_size, 1))
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss={"output_gender": "binary_crossentropy", "output_age": "mae"},
        metrics={"output_gender": "accuracy", "output_age": "mae"},
        loss_weights={"output_gender": 1.0, "output_age": 1.0},
    )

    checkpoint_path = os.path.join(os.path.dirname(args.output), "best_model.keras")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
    print("Training complete.")

    print("Evaluating on test set...")
    model.load_weights(checkpoint_path)
    test_results = model.evaluate(test_ds, verbose=1)
    print(f"Model Metrics Names: {model.metrics_names}")
    print(f"Test results: {test_results}")

    os.replace(checkpoint_path, args.output)
    print(f"Final model saved to: {args.output}")


if __name__ == "__main__":
    main()
