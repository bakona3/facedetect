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
