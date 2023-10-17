# Hey, I'm trying to work with some cool stuff here!

# First, I'm importing some libraries I need. They're like tools for my project.
import os  # I think this is for working with files and folders.
import cv2  # OpenCV is great for pictures and videos.
import numpy as np  # This is for math and numbers.
import xml.etree.ElementTree as ET  # XML, I think it's for saving data.

# Now, I'm telling my code where to find the calibration data.
root_directory = "Wildtrack/calibrations"

# I need to make lists to hold the data for each camera.
intrinsic_original_data = []  # This one for the original distortion.
intrinsic_zero_data = []  # This is for zero distortion.
extrinsic_data = []  # And this for the camera's position and orientation.

# I'm creating folders where the different data types are stored.
extrinsic_dir = os.path.join(root_directory, "extrinsic")  # Extrinsic data folder.
intrinsic_original_dir = os.path.join(root_directory, "intrinsic_original")  # Original distortion folder.
intrinsic_zero_dir = os.path.join(root_directory, "intrinsic_zero")  # Zero distortion folder.

# Here, I'm writing a function that loads data from a folder.
def load_calibration_data(calibration_dir):
    calibration_data = []  # This is where I'm going to put the data.
    for file_name in os.listdir(calibration_dir):
        if file_name.endswith(".xml"):  # I only want XML files.
            file_path = os.path.join(calibration_dir, file_name)
            calibration_data.append(cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ))
    return calibration_data

# This function is almost the same, but for the extrinsic data.
def load_extrinsic_calibration_data(calibration_dir):
    calibration_data = []  # Another place for data.
    for file_name in os.listdir(calibration_dir):
        if file_name.endswith(".xml"):
            file_path = os.path.join(calibration_dir, file_name)
            calibration_data.append(file_path)  # Just saving file paths.
    return calibration_data

# Now, I'm loading the data for real.
extrinsic_data = load_extrinsic_calibration_data(extrinsic_dir)  # Loading the extrinsic data.
intrinsic_original_data = load_calibration_data(intrinsic_original_dir)  # Loading original distortion data.
intrinsic_zero_data = load_calibration_data(intrinsic_zero_dir)  # And zero distortion data.

# I'm getting ready to save camera-specific stuff.
camera_matrices = []  # I think this is for the camera's properties.
dist_coeffs_original = []  # This is distortion for the original data.
dist_coeffs_zero = []  # And this is for the zero distortion.
rotation_vectors = []  # These are like angles.
translation_vectors = []  # These are for positions.

# Time to load data for all the cameras.
for i in range(len(extrinsic_data)):
    # First, I'm grabbing the camera matrix for the original distortion.
    camera_matrix = intrinsic_original_data[i].getNode('camera_matrix').mat()
    camera_matrices.append(camera_matrix)

    # Now, I'm taking the distortion coefficients for the original data.
    dist_coeff_original = intrinsic_original_data[i].getNode('distortion_coefficients').mat()
    dist_coeffs_original.append(dist_coeff_original)

    # Same thing, but for zero distortion.
    dist_coeff_zero = intrinsic_zero_data[i].getNode('distortion_coefficients').mat()
    dist_coeffs_zero.append(dist_coeff_zero)

    # Now, I'm dealing with extrinsic data, like the camera's position and orientation.
    extrinsic_tree = ET.parse(os.path.join(extrinsic_data[i]))  # I'm reading an XML file.
    extrinsic_root = extrinsic_tree.getroot()  # I think this is the main part of the XML.

    # I'm getting the rotation and translation vectors.
    rvec = [float(x) for x in extrinsic_root.find('rvec').text.split()]  # They're in the XML.
    tvec = [float(x) for x in extrinsic_root.find('tvec').text.split()]  # Splitting and converting to numbers.

    # I'm saving those vectors.
    rotation_vectors.append(rvec)
    translation_vectors.append(tvec)

# Finally, I'm going to show all the data I collected.
print("Camera Matrices:")  # Printing camera properties.
print(camera_matrices)
print("\nDistortion Coefficients (Original):")  # Printing original distortion data.
print(dist_coeffs_original)
print("\nDistortion Coefficients (Zero Distortion):")  # Printing zero distortion data.
print(dist_coeffs_zero)
print("\nRotation Vectors (rvec):")  # Printing rotation angles.
print(rotation_vectors)
print("\nTranslation Vectors (tvec):")  # And the camera positions.
print(translation_vectors)
