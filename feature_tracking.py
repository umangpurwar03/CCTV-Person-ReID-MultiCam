# First, I'm importing some libraries. These are like toolkits for my project.
import os  # This one is for working with files and folders.
import cv2  # OpenCV is great for handling images and videos.
import numpy as np  # It helps with mathematical operations.
import xml.etree.ElementTree as ET  # XML handling for data.
from ultralytics import YOLO  # This is for object detection.
from deep_sort.utils.parser import get_config  # More stuff for object tracking.
from deep_sort.deep_sort import DeepSort  # Helps with object tracking.
from deep_sort.sort.tracker import Tracker  # More tracking stuff.
import torch  # This is for using a GPU if available.
from feature_helping_function import extract_cnn_embeddings,extract_color_histograms
import json

# Let's see if we have a GPU. If we do, we'll use it; otherwise, we'll use the CPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Here I'm getting a pre-trained object tracking model.
deep_sort_weights = 'deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path=deep_sort_weights, max_age=90)

# I've set the root directory for my calibration data.
root_directory = "Wildtrack/calibrations"

# I'm preparing lists to store data for each camera.
intrinsic_original_data = []  # Data with original distortion.
intrinsic_zero_data = []  # Data with zero distortion.
extrinsic_data = []  # Information about camera position and orientation.

# Now, I'm figuring out the subdirectories for different types of calibration data.
extrinsic_dir = os.path.join(root_directory, "extrinsic")  # Extrinsic data directory.
intrinsic_original_dir = os.path.join(root_directory, "intrinsic_original")  # Original distortion directory.
intrinsic_zero_dir = os.path.join(root_directory, "intrinsic_zero")  # Zero distortion directory.

# This function helps me load calibration data from a directory.
def load_calibration_data(calibration_dir):
    calibration_data = []  # I'll keep the data in this list.
    for file_name in os.listdir(calibration_dir):
        if file_name.endswith(".xml"):  # Only consider XML files.
            file_path = os.path.join(calibration_dir, file_name)
            calibration_data.append(cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ))
    return calibration_data

# This function is similar, but it loads extrinsic calibration data.
def load_extrinsic_calibration_data(calibration_dir):
    calibration_data = []  # Another list for the paths to the extrinsic data files.
    for file_name in os.listdir(calibration_dir):
        if file_name.endswith(".xml"):  # Again, just consider XML files.
            file_path = os.path.join(calibration_dir, file_name)
            calibration_data.append(file_path)  # I'm saving the file paths.
    return calibration_data

# Now, I'm loading data for each type of calibration.
extrinsic_data = load_extrinsic_calibration_data(extrinsic_dir)  # Loading extrinsic data file paths.
intrinsic_original_data = load_calibration_data(intrinsic_original_dir)  # Loading data with original distortion.
intrinsic_zero_data = load_calibration_data(intrinsic_zero_dir)  # Loading data with zero distortion.

# I'm creating lists to store camera-specific parameters.
camera_matrices = []  # These are properties related to the camera.
dist_coeffs_original = []  # Distortion coefficients for the data with original distortion.
dist_coeffs_zero = []  # Distortion coefficients for the data with zero distortion.
rotation_vectors = []  # These are like rotation angles.
translation_vectors = []  # And these are for position.

# Now, I'm going to load calibration data for all cameras.
for i in range(len(extrinsic_data)):
    # I'm getting the camera matrix for intrinsic calibration with original distortion.
    camera_matrix = intrinsic_original_data[i].getNode('camera_matrix').mat()
    camera_matrices.append(camera_matrix)

    # And I'm loading the distortion coefficients for intrinsic calibration with original distortion.
    dist_coeff_original = intrinsic_original_data[i].getNode('distortion_coefficients').mat()
    dist_coeffs_original.append(dist_coeff_original)

    # The same thing for intrinsic calibration with zero distortion.
    dist_coeff_zero = intrinsic_zero_data[i].getNode('distortion_coefficients').mat()
    dist_coeffs_zero.append(dist_coeff_zero)

    # Now, I'm loading and parsing an XML file for the extrinsic calibration.
    extrinsic_tree = ET.parse(os.path.join(extrinsic_data[i]))  # Reading an XML file.
    extrinsic_root = extrinsic_tree.getroot()  # I think this is the main part of the XML.

    # I'm extracting rotation vectors (rvec) and translation vectors (tvec).
    rvec = [float(x) for x in extrinsic_root.find('rvec').text.split()]  # Splitting and converting to numbers.
    tvec = [float(x) for x in extrinsic_root.find('tvec').text.split()]

    # Saving these vectors.
    rotation_vectors.append(rvec)
    translation_vectors.append(tvec)

# Now, I'm doing object detection using the YOLO model.
model = YOLO("yolov8m.pt")  # Loading the YOLO model.

# I've set up the folders for input and output videos.
video_folder = r"Wildtrack"  # Where my video files are.
output_folder = r"output"  # Where I'll save the output videos.

# I'm listing all the video files in the input folder.
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
print(video_files)  # Just checking which videos I'm going to process.

individual_features = []

# I'm going to loop through each video and process it.
for i, video_file in enumerate(video_files):
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)  # Capturing frames from the video.

    # To save the output video, I need to set up a codec and a VideoWriter.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # I think this sets the video codec.
    output_video_path = os.path.join(output_folder, f"output_{i}.mp4")  # The path for the output video.

    # I'm getting the width and height of the video frames.
    frame_width = int(cap.get(3))  # I think this is the frame width.
    frame_height = int(cap.get(4))  # And this is the frame height.

    # Creating a VideoWriter to save the output video.
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))

    # Now, I'm going to process each frame in the video.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # I'll stop if there are no more frames.

        # I'm undistorting the frame using camera calibration data.
        undistorted_frame = cv2.undistort(frame, camera_matrices[i], dist_coeffs_original[i])

        # Here, I'm doing object detection using the YOLO model on the undistorted frame.
        results = model(undistorted_frame, classes=0, conf=0.5)  # Detecting objects.

        # I'm defining class names for objects I'm interested in, like "person."
        class_names = ["person"]

        # Now, I'm going through the detected results.
        for result in results:
            boxes = result.boxes  # Bounding boxes for the objects.
            probs = result.probs  # Probabilities for object classes.
            cls = boxes.cls.tolist()  # Converting class information to a list.
            xyxy = boxes.xyxy  # I think this is the format of the bounding boxes.
            conf = boxes.conf
            xywh = boxes.xywh  # Bounding boxes with xywh format.

            # Going through each class found in the frame.
            for class_index in cls:
                class_name = class_names[int(class_index)]  # Converting class index to a name.

        # I'm converting some data to numpy arrays for further processing.
        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)

        # Feature extraction with CNN embeddings
        embeddings = extract_cnn_embeddings(model, undistorted_frame, xyxy)

         # Feature extraction with color histograms
        histograms = extract_color_histograms(undistorted_frame, xyxy)

        # Here comes the object tracking part.
        tracks = tracker.update(bboxes_xywh, conf, undistorted_frame)

        # I'm going through each tracked object.
        for track in tracker.tracker.tracks:
            track_id = track.track_id  # I'm getting the object's track ID.

                    # Store the features and associated information in a dictionary
            individual_info = {
                "track_id": track_id,
                "embeddings": embeddings,  # Replace with the actual embeddings
                "histograms": histograms # Replace with the actual histograms
            }

            individual_features.append(individual_info)

            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # These are bounding box coordinates.
            w = x2 - x1  # Calculating width.
            h = y2 - y1  # Calculating height.

            # Defining some colors for drawing bounding boxes.
            red_color = (0, 0, 255)  # Red color in BGR format.
            blue_color = (255, 0, 0)  # Blue color.
            green_color = (0, 255, 0)  # Green color.

            # Picking a color based on the track ID.
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
            elif color_id == 1:
                color = blue_color
            else:
                color = green_color

            # Drawing a bounding box around the object.
            cv2.rectangle(undistorted_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            # Adding a text label with the class name and track ID.
            text_color = (0, 0, 0)  # Black color for the text.
            cv2.putText(undistorted_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            out.write(undistorted_frame)  # Saving the frame with bounding boxes and labels.

            # Showing the frame.
            cv2.imshow("Video", undistorted_frame)

            # I'm waiting for a key press to exit. Press 'q' to stop the processing.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()  # Releasing the video capture object.
cv2.destroyAllWindows()  # Closing all OpenCV windows.

# Define the JSON file path where you want to save the features
json_file_path = "individual_features.json"

# Save the features to the JSON file
with open(json_file_path, "w") as json_file:
    json.dump(individual_features, json_file)