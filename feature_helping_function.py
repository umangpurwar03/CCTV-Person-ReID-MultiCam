# Function to extract CNN embeddings
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
import torch
import cv2
import numpy as np
# Define the device for GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Function to extract CNN embeddings with input normalization
def extract_cnn_embeddings(model, frame, boxes):
    # Crop and resize the regions of interest (ROIs) from the frame
    cropped_rois = []
    for box in boxes:
        x1, y1, x2, y2 = box
        roi = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_rois.append(roi)

    # Preprocess the ROIs for the model
    transformations = transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)),
                                          transforms.ToTensor()])
    processed_rois = [transformations(roi) / 255.0 for roi in cropped_rois]  # Normalize by dividing by 255

    # Convert the processed ROIs to a batch
    batch = torch.stack(processed_rois)

    # Pass the batch through the model to get embeddings
    with torch.no_grad():
        embeddings = model(batch.to(device))

    return embeddings


# Function to extract color histograms
def extract_color_histograms(frame, boxes):
    histograms = []

    for box in boxes:
        x1, y1, x2, y2 = box
        roi = frame[int(y1):int(y2), int(x1):int(x2)]

        # Convert the ROI to HSV color space
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Calculate the histogram
        hist_hue = cv2.calcHist([hsv_roi], [0], None, [256], [0, 256])
        hist_saturation = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
        hist_value = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])

        # Normalize the histograms
        hist_hue = cv2.normalize(hist_hue, hist_hue, 0, 1, cv2.NORM_MINMAX)
        hist_saturation = cv2.normalize(hist_saturation, hist_saturation, 0, 1, cv2.NORM_MINMAX)
        hist_value = cv2.normalize(hist_value, hist_value, 0, 1, cv2.NORM_MINMAX)

        # Concatenate the histograms
        hist_combined = np.concatenate((hist_hue, hist_saturation, hist_value))

        histograms.append(hist_combined)

    return histograms
