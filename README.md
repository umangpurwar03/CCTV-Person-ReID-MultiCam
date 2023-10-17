# Computer Vision Assignment - Person Re-Identification

## Objective:
The objective of this assignment is to assess your computer vision skills in person re-identification using publicly available CCTV footage. You will develop a model to identify and track individuals across multiple camera views.

## Requirements:
- Dataset I have used for [**`dataset`**](https://academictorrents.com/details/5931991ad96a83cca85c0604061e766abefdf94b)
- Use Python and popular computer vision libraries like OpenCV and PyTorch.

## Instructions:

### Camera Callibration
- [**`calibration.py`**](calibration.py): The openCv is used to calibrate all the camera vedios present in Wildtrack dataset

### Step 2: Person Detection and Tracking
- `person_detection`: For detecting Person i have used YOLO8m model
- `tracking_algorithm`: Deep SHORT tracking algorithm to track individuals across frames and camera views.
-[**`track.py`**](track.py) you can i find the python code over there

### Feature Extraction 
- [**`feature_helping_function.py`**](feature_helping_function.py):In this i have defined function to extract features
- [**`feature_tracking.py`**](feature_tracking.py): In this i have collaborated all the camera callibration tracking detection and feature extraction.
