import tensorflow as tf
import numpy as np
import cv2
import json


model = tf.keras.models.load_model("plant_model.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

with open("plantinfo.json", "r") as f:
    plant_info = json.load(f)

image_path = r"C:\Users\varsh\OneDrive\Desktop\Mini Project\test\Jatropha\Bright-pink-Jatropha-flowers-with-yellow-centers-blooming-in-Dong-Nai-Vietnam-Perfect-for-nature-and-tropical-themes-Stock-Photo_jpg.rf.68a934fa332abaae3c4682c0681f9633.jpg"
img = cv2.imread(image_path)

if img is None:
    print("❌ Image not found")
    exit()

img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
class_index = int(np.argmax(prediction))
confidence = float(np.max(prediction))

plant_name = class_names[str(class_index)]
THRESHOLD = 0.65

if confidence < THRESHOLD:
    print("⚠️ Plant not recognized (unknown plant)")
else:
    print("🌱 Plant Name:", plant_name)
    print("📊 Confidence:", round(confidence * 100, 2), "%")
    info = plant_info.get(plant_name)
    if info:
        if info["toxic"]:
            print("☠️ Toxicity: POISONOUS")
        else:
            print("✅ Toxicity: NON-POISONOUS")

        print("💊 Remedy / Action:", info["remedy"])
    else:
        print("ℹ️ No toxicity data available for this plant")
