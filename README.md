# ```                AI & ML Project                ```

# Hand Gesture Recognition using TensorFlow Object Detection API

This repository contains a Python script for real-time hand gesture recognition using TensorFlow Object Detection API. The script uses a pre-trained model for object detection to identify and visualize hand gestures in a live video stream.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python
- TensorFlow
- OpenCV
- TensorFlow Object Detection API

You can install the required Python packages using the following command:

```bash
pip install tensorflow opencv-python
```

For TensorFlow Object Detection API installation, please refer to the official documentation: [TensorFlow Object Detection API Installation](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

## Usage

1. Clone the repository:

```bash
git clone https://github.com/NoorMahammad-S/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Update the paths to the pre-trained model and label map in the script (`hand_gesture_recognition.py`). Replace the placeholders in the following lines with your actual paths:

```python
# Path to the pre-trained model and label map
PATH_TO_MODEL = 'path/to/your/pretrained/model/saved_model'
PATH_TO_LABELS = 'path/to/your/label/map.pbtxt'
```

3. Run the script:

```bash
python hand_gesture_recognition.py
```

4. Press 'q' to exit the video stream.

## Customization

- Adjust the `min_score_thresh` parameter in the script to control the confidence threshold for detection results.

```python
min_score_thresh=0.60  # Adjust this threshold based on your needs
```

Feel free to customize the script according to your requirements.

## Acknowledgments

- This project is based on the TensorFlow Object Detection API. Check out the official documentation for more details: [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

