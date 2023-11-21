import cv2
import tensorflow as tf

# Load the pre-trained model
model = tf.saved_model.load('path/to/your/pretrained/model')

# Load the label map
category_index = {1: {'id': 1, 'name': 'hand'}}

# Set up the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make the prediction
    input_tensor = tf.convert_to_tensor([rgb_frame])
    detections = model(input_tensor)

    # Process the detections
    for detection in detections['detection_boxes'][0].numpy():
        ymin, xmin, ymax, xmax = detection
        h, w, _ = frame.shape
        xmin = int(xmin * w)
        xmax = int(xmax * w)
        ymin = int(ymin * h)
        ymax = int(ymax * h)

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
