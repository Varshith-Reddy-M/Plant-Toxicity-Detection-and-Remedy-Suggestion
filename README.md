# 🌿Plant Toxicity Detection and Remedy Suggestion

A deep learning-based plant identification system that classifies plants from images and provides toxicity information along with first-aid remedies. Built using MobileNetV2 transfer learning with TensorFlow/Keras.

---
## 🚀 Features

🧠 Transfer learning using MobileNetV2 (ImageNet Pretrained)

🔄 Two-phase training (Feature Extraction + Fine-Tuning)

📊 Confidence threshold for unknown plant detection

☠️ Toxicity detection with remedy suggestions

⚡ Lightweight and edge-device friendly

---

## Supported Plants

| Plant | Toxic |
|---|---|
| Ageratum | Yes |
| Argemone | Yes |
| Calotropis Gigantea | Yes |
| Castor | Yes |
| Centella Asiatica | No |
| Croton Shrub | Yes |
| Datura | Yes |
| Euphorbia | Yes |
| Jatropha | Yes |
| Lantana Camara | Yes |
| Mimosa Pudica | No |
| Money Plant | Yes |
| Ocimum Sanctum | No |
| Oleander | Yes |
| Tridax | No |

---

## ☠️Toxicity & Remedy Information

Toxicity information is stored in:

```plantinfo.json```

Example Structure:
```
{
  "Datura": {
    "toxic": true,
    "remedy": "No home remedy. Seek immediate medical attention."
  }
}
```

## Project Structure

```
├── Dataset               # Training dataset
├── test                  # Testing dataset
├── valid                 # Validation dataset for fine tuning
├── plant_model.h5        # Trained Keras model
├── class_names.json      # Index-to-class name mapping
├── plantinfo.json        # Toxicity and remedy data per plant
├── trainmodel.py         # Model training script
├── testmodel.py          # Inference / prediction script
├── training_graph.png    # Accuracy and Loss graph
└── README.md
```

---

## Requirements

```
Python 3.8+
TensorFlow 2.x
OpenCV
NumPy
```

---

## Installation

```bash
git clone https://github.com/your-username/plant-toxicity-detector.git
cd plant-toxicity-detector
pip install tensorflow opencv-python numpy
```

---

## Usage

1. Open `testmodel.py` and set your image path:

```python
image_path = "path/to/your/plant_image.jpg"
```

2. Run:

```bash
python testmodel.py
```

---

## Example Output

```
🌱 Plant Name: Oleander
📊 Confidence: 91.43 %
☠️ Toxicity: POISONOUS
💊 Remedy / Action: Immediate first aid. Rinse mouth properly. Drink water. Emergency care required
```

If confidence is below 65%:

```
⚠️ Plant not recognized (unknown plant)
```

---

## Model Architecture

```
Base:      MobileNetV2 (pretrained on ImageNet)
Input:     224 x 224 x 3
Head:      GlobalAveragePooling2D
           Dense(256, activation='relu')
           Dropout(0.4)
           Dense(128, activation='relu')
           Dropout(0.3)
           Dense(64, activation='relu')
           Dropout(0.2)
           Dense(15, activation='softmax')
Threshold: 0.65
```

---

## Training

```
Phase 1 — Top layers only (base frozen)
  Optimizer : Adam lr=0.0001
  Epochs    : 20

Phase 2 — Fine-tuning (last 30 MobileNetV2 layers unfrozen)
  Optimizer : Adam lr=0.00001
  Epochs    : 20

Augmentation: rotation=25, zoom=0.2, horizontal_flip=True
```

Dataset folder structure required:

```
Dataset/
  Ageratum/
  Argemone/
  ...

valid/
  Ageratum/
  Argemone/
  ...
```

Run training:

```bash
python trainmodel.py
```

---

## 🎯Applications

Trekking safety assistant

Educational plant identification tool

Toxic plant awareness system

---

## 📈Future Improvements

Increase number of plant species

Expand dataset size

Embedded AI deployment (Raspberry Pi)

Add real-time camera integration

---

## Disclaimer

This tool is for educational purposes only. In any case of suspected plant poisoning, contact emergency services immediately. Do not rely solely on this application for medical decisions.
