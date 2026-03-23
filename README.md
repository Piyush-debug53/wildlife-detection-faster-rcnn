# Wildlife Detection using Faster R-CNN

> ⭐ If you like this project, consider giving it a star!

---

## Project Overview

This project implements a **wildlife object detection system using Faster R-CNN**, focusing on accurate detection of animals in real-world scenarios such as forests, railway tracks, and wildlife sanctuaries.

It is part of a **comparative study with YOLOv8**, where both models are trained on the **same dataset** to analyze the trade-off between **speed (YOLO)** and **accuracy (Faster R-CNN)**.

---

## Problem Statement

Wildlife movement near human infrastructure (like roads and railway tracks) poses risks to both animals and humans. Traditional monitoring systems lack automation and accuracy.

This project aims to:

* Detect animals in images and videos
* Provide bounding boxes with class labels
* Enable future real-time deployment for safety systems

---

## Dataset

The same dataset was used for both **YOLOv8 and Faster R-CNN** to ensure a **fair comparison**.

* Original format: YOLO (`.txt` labels)
* Converted format: COCO (`.json` annotations)

    **Download full dataset:**
👉 [Google Drive Link](https://drive.google.com/file/d/1dRf1TJCpRDk9r1wVF9HN4N0NuCwbtZol/view?usp=sharing)

> Note: Only a **sample dataset** is included in this repository which already includes annotations.json file for train, val and test.

---

## CLASSES

```markdown
- Bear
- Deer
- Dog 
- Tiger  
- Elephant  

## Annotation Conversion

The dataset was originally in YOLO format.
For Faster R-CNN compatibility, annotations were converted into **COCO format** using:

```bash
scripts/coco_conversion.py
```

---

## Tech Stack

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy

---

## Model Details

* Model: Faster R-CNN
* Backbone: ResNet (Torchvision)
* Type: Two-stage detector

---

## Project Structure

```text
wildlife-detection-faster-rcnn/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── train.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── dataset.py
│   └── webcam.py
│
├── scripts/
│   └── coco_conversion.py
│
├── Sample_Dataset/
│   ├── images/
│   ├── labels/
│   └── annotations.json
│
├── Inputs/
│   └── images/
│
├── outputs/
│   └── images/
│
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│
└── report/
    └── comparison_report.pdf
```

---

## How to Run

### 1️ Clone the repository

```bash
git clone https://github.com/Piyush-debug53/wildlife-detection-faster-rcnn.git
cd wildlife-detection-faster-rcnn
```

---

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️ Run inference

```bash
python src/inference.py
```

---

## Requirements

All dependencies are listed in `requirements.txt`.

Install using:

```bash
pip install -r requirements.txt
```

## Demo

### Sample Detection

![Detection](outputs/images/bear (1000).jpg)
![Detection](outputs/images/bear (1001).jpg)
![Detection](outputs/images/bear (1002).jpg)

---

## Results

The model was evaluated using standard object detection metrics.

* **mAP@50:** ~
* Confusion Matrix and evaluation plots are included in the `results/` folder

### Visual Results

* Confusion Matrix
* Precision-Recall Curves
* Detection Outputs

---

## Key Insights

* Faster R-CNN provides **higher accuracy** but slower inference compared to YOLOv8
* Suitable for applications where **precision is more important than speed**
* COCO format enables structured dataset handling

---

## Comparison with YOLOv8

| Feature      | YOLOv8      | Faster R-CNN   |
| ------------ | ----------- | -------------- |
| Speed        | Very Fast   | Slower         |
| Accuracy     | Good        | Higher         |
| Use Case     | Real-time   | High precision |
| Architecture | One-stage   | Two-stage      |

---

## Report

A detailed comparative report between YOLOv8 and Faster R-CNN will be included soon.

© Piyush Bhatia, 2026
For academic and portfolio use only.
Unauthorized reproduction or submission is prohibited.

---

## Future Work

* Real-time deployment using camera feed
* Edge device integration (Jetson Nano / ESP32-CAM)
* Model optimization for speed

---

## Author

**Piyush Bhatia**

* GitHub: https://github.com/Piyush-debug53
* LinkedIn: https://www.linkedin.com/in/piyush-bhatia-14274a28a/

---
