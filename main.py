import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Path to the pre-trained model and label map
PATH_TO_MODEL = 'path/to/your/pretrained/model/saved_model'
PATH_TO_LABELS = 'path/to/your/label/map.pbtxt'

# Load the pre-trained model
detect_fn = tf.saved_model.load(PATH_TO_MODEL)

# Load the label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Set up the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Make the prediction using the pre-trained model
    input_tensor = tf.convert_to_tensor([rgb_frame])
    detections = detect_fn(input_tensor)

    # Visualize the detections on the frame
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(detections['detection_boxes']),
        np.squeeze(detections['detection_classes']).astype(np.int32),
        np.squeeze(detections['detection_scores']),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60  # Adjust this threshold based on your needs
    )

    # Display the resulting frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
