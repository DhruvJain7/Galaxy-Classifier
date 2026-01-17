---
title: Galaxy Classifier
emoji: 🌌
colorFrom: blue
colorTo: black
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---
# 🌌 Galaxy Morphology Classifier

An interactive web application that uses Deep Learning to classify galaxies into 10 morphological categories based on the **Galaxy10 DECaLS** dataset.

## 🚀 The Prototype
This application is powered by a fine-tuned **EfficientNetV2** model. It doesn't just provide a label; it provides a detailed "Galaxy Profile" including:
* **Morphological Description:** Detailed breakdown of visual features.
* **Scientific Facts:** Interesting astronomical context for each type.
* **Dataset Rarity:** How often this type appears in the DECaLS survey.

## 📊 Performance
* **Accuracy:** ~82%
* **Architecture:** EfficientNetV2-B2 (Fine-tuned via 3-phase stabilization)
* **Dataset:** 17,736 images from the DESI Legacy Imaging Surveys.

## 🛠️ How to Run Locally
1. Clone the repo: `git clone https://github.com/YOUR_USERNAME/Galaxy-Classifier-App.git`
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `python app.py`

> **Note:** Ensure you have the `best_galaxy_classifier.keras` model file in the specified directory.
