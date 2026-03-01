import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
import json

# =========================
# SETTINGS
# =========================
DATASET_PATH = "Dataset"
VAL_PATH = "valid"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# =========================
# LOAD DATASET
# =========================
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    VAL_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Save class names
class_indices = train_data.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

with open("class_names.json", "w") as f:
    json.dump(index_to_class, f)

print("Saved class names:", index_to_class)

# =========================
# LOAD BASE MODEL
# =========================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# =========================
# ADD CUSTOM CLASSIFIER
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# PHASE 1: TRAIN TOP LAYERS
# =========================
print("\n--- Phase 1: Training top layers ---\n")

base_model.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# =========================
# PHASE 2: FINE-TUNING
# =========================
print("\n--- Phase 2: Fine-tuning top MobileNet layers ---\n")

base_model.trainable = True

# Freeze early layers, unfreeze last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save("plant_model.h5")

print("\n✅ Training Complete")
print("📁 Model saved as plant_model.h5")