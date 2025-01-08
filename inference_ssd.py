import cv2
import numpy as np
import tensorflow as tf

def load_ssd_model(saved_model_dir):
    model = tf.saved_model.load(saved_model_dir)
    return model

def detect_objects_ssd(model, image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

if __name__ == "__main__":
    # Path to SSD model
    saved_model_dir = "ssd_model"  

    # Load model
    model = load_ssd_model(saved_model_dir)

    # Real-time inference with webcam
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detect_objects_ssd(model, frame)

     

        # Display
        cv2.imshow("SSD Detection", frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
