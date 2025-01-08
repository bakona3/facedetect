# real_time_detection.py
"""
Unified real-time detection script that supports both YOLO and SSD models.
"""
import cv2
import torch
import numpy as np
import tensorflow as tf

# Hardcoded paths
YOLO_WEIGHTS_PATH = "best.pt"  # Path to YOLO weights file
SSD_MODEL_DIR = "ssd_model"    # Path to SSD saved model directory

def load_yolo_model(weights_path):
    """Load YOLO model from the given weights path."""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

def load_ssd_model(saved_model_dir):
    """Load SSD model from the given saved model directory."""
    model = tf.saved_model.load(saved_model_dir)
    return model

def detect_objects_yolo(model, image):
    """Run YOLO inference on the input image."""
    results = model(image)
    return results

def detect_objects_ssd(model, image):
    """Run SSD inference on the input image."""
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def visualize_yolo_results(image, results):
    """Visualize YOLO results by rendering bounding boxes and labels."""
    annotated_frame = np.array(results.render()[0])
    return annotated_frame

def visualize_ssd_results(image, detections):
    """Visualize SSD results by drawing bounding boxes and labels."""
    # Extract detection results
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # Get image dimensions
    height, width, _ = image.shape

    # Draw bounding boxes and labels
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)

            # Draw rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Add label
            label = f"Class: {classes[i]}, Score: {scores[i]:.2f}"
            cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def main(model_type):
    # Load the selected model
    if model_type == "yolo":
        print("Loading YOLO model...")
        model = load_yolo_model(YOLO_WEIGHTS_PATH)
    elif model_type == "ssd":
        print("Loading SSD model...")
        model = load_ssd_model(SSD_MODEL_DIR)
    else:
        raise ValueError("Invalid model type. Choose 'yolo' or 'ssd'.")

    # Start real-time detection with webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        if model_type == "yolo":
            results = detect_objects_yolo(model, frame)
            output_frame = visualize_yolo_results(frame, results)
        elif model_type == "ssd":
            detections = detect_objects_ssd(model, frame)
            output_frame = visualize_ssd_results(frame, detections)

        # Display the output frame
        cv2.imshow("Real-Time Detection", output_frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose the model type: "yolo" or "ssd"
    MODEL_TYPE = "yolo"  # Change to "ssd" if you want to use the SSD model

    # Run real-time detection
    main(MODEL_TYPE)