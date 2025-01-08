import torch
import cv2
import numpy as np

def load_yolo_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

def detect_objects_yolo(model, image):
    results = model(image)
    return results

def visualize_results(image, results):
    annotated_frame = np.array(results.render()[0])
    return annotated_frame

if __name__ == "__main__":
    # Path to YOLO weights
    weights_path = "best.pt"  

    # Load model
    model = load_yolo_model(weights_path)

    # Real-time inference with webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        results = detect_objects_yolo(model, frame)

        # Visualize
        output_frame = visualize_results(frame, results)

        # Display
        cv2.imshow("YOLO Detection", output_frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
