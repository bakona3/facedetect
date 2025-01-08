# Face Mask Detection Project

This project is designed to detect whether a person is wearing a face mask, not wearing a mask, or wearing a mask incorrectly in real-time using a webcam. It supports two deep learning models: **YOLO (You Only Look Once)** and **SSD (Single Shot Detector)**.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)

---

## Project Overview

The goal of this project is to provide a real-time face mask detection system that can be used in various environments, such as public spaces, offices, or schools, to ensure compliance with mask-wearing guidelines. The system uses deep learning models to classify and detect faces into three categories:
- **With Mask**
- **Without Mask**
- **Mask Worn Incorrectly**

---

## Features

- **Real-Time Detection**: Captures video from a webcam and performs real-time inference.
- **Multiple Models**: Supports both YOLO and SSD models for detection.
- **Visualization**: Displays bounding boxes and labels for detected objects.
- **Easy to Use**: Simple setup and execution with clear instructions.

---

## Requirements

To run this project, you need the following dependencies:

- Python 3.8 or higher
- OpenCV (`opencv-python`)
- PyTorch (`torch` and `torchvision`)
- TensorFlow (`tensorflow`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`) (for evaluation scripts)
- Seaborn (`seaborn`) (for evaluation scripts)

You can install the required libraries using the following command:

```bash
pip install torch torchvision opencv-python tensorflow numpy matplotlib seaborn
```

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
   ```

2. **Download Pretrained Models**:
   - For YOLO: Download the `.pt` weights file (e.g., `best.pt`) and place it in the project directory.
   - For SSD: Download the TensorFlow SavedModel directory (e.g., `ssd_model`) and place it in the project directory.

3. **Run the Script**:
   - Modify the `MODEL_TYPE` variable in `real_time_detection.py` to `"yolo"` or `"ssd"` depending on the model you want to use.
   - Run the script:
     ```bash
     python real_time_detection.py
     ```

---

## Usage

### Real-Time Detection
1. **YOLO Model**:
   - Set `MODEL_TYPE = "yolo"` in `real_time_detection.py`.
   - Run the script:
     ```bash
     python real_time_detection.py
     ```

2. **SSD Model**:
   - Set `MODEL_TYPE = "ssd"` in `real_time_detection.py`.
   - Run the script:
     ```bash
     python real_time_detection.py
     ```

### Evaluation
- Use the `evaluation.py` script to plot the confusion matrix and F1-Confidence curve for model evaluation.
  ```bash
  python evaluation.py
  ```

---

## Project Structure

Here’s the structure of the project:

```
face-mask-detection/
├── best.pt                  # YOLO model weights
├── ssd_model/               # SSD saved model directory
├── real_time_detection.py   # Main script for real-time detection
├── inference_yolo.py        # YOLO inference script
├── inference_ssd.py         # SSD inference script
├── evaluation.py            # Script for model evaluation
├── README.md                # Project documentation
└── requirements.txt         # List of dependencies
```

---

## Results

### Real-Time Detection
- The system displays a live video feed with bounding boxes and labels for detected objects.
- Example output:
  - **With Mask**: Green bounding box with label "With Mask".
  - **Without Mask**: Red bounding box with label "Without Mask".
  - **Mask Worn Incorrectly**: Yellow bounding box with label "Mask Incorrect".

### Evaluation Metrics
- Confusion Matrix: Visualizes the performance of the model across all classes.
- F1-Confidence Curve: Shows the trade-off between precision and recall for different confidence thresholds.

---


## Acknowledgments

- **YOLOv5**: The YOLO model used in this project is based on the Ultralytics implementation of YOLOv5.
- **TensorFlow Model Zoo**: The SSD model is based on pretrained models from the TensorFlow Model Zoo.

