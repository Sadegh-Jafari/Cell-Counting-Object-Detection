import os
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLO model
model_path = '/work/hdd/bczm/sjafarisheshtamad/COMS571/runs/detect/train59/weights/best.pt'
model = YOLO(model_path)

# Paths to images and ground truth folders
images_folder = '/work/hdd/bczm/sjafarisheshtamad/COMS571/IDCIA v2/images'
ground_truth_folder = '/work/hdd/bczm/sjafarisheshtamad/COMS571/IDCIA v2/ground_truth'

# Sharpening and deblurring parameters
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# Get the list of all image files
image_files = sorted(os.listdir(images_folder))

# Initialize variables for ACP calculation
within_margin_count = 0  # Number of predictions within 5% of ground truth
image_count = 0  # Total number of processed images

# Process each image and its corresponding ground truth CSV
for image_file in image_files:
    # Skip hidden files or invalid files
    if image_file.startswith('.'):
        continue
    
    # Load the image
    image_path = os.path.join(images_folder, image_file)
    img = cv2.imread(image_path)
    
    # Apply sharpening filter
    sharpened = cv2.filter2D(img, -1, kernel)

    # Apply deblurring (using Gaussian blur)
    deblurred = cv2.GaussianBlur(sharpened, (0, 0), 3)
    enhanced_img = cv2.addWeighted(sharpened, 3, deblurred, -1, 0)

    # Corresponding CSV file (assuming same base name as the image file)
    base_name = os.path.splitext(image_file)[0]
    csv_file = os.path.join(ground_truth_folder, f"{base_name}.csv")
    
    # Check if the ground truth CSV exists
    if not os.path.exists(csv_file):
        print(f"Ground truth file not found for image: {image_file}")
        continue
    
    # Load ground truth data (x, y coordinates)
    ground_truth_data = pd.read_csv(csv_file)
    
    # Count the number of cells in the ground truth (each row is a cell)
    ground_truth_cells = len(ground_truth_data)
    
    # Run YOLOv8 detection on the enhanced image with lower confidence threshold
    results = model(enhanced_img, conf=0.25, max_det=1000)  # Set confidence threshold to 0.2
    
    # Count the number of detected cells by YOLOv8
    detected_cells = len(results[0].boxes)
    
    # Avoid division by zero for ACP calculation
    if ground_truth_cells > 0:
        error_margin = 0.05 * ground_truth_cells  # 5% of ground truth
        if abs(detected_cells - ground_truth_cells) <= error_margin:
            within_margin_count += 1  # Count predictions within the acceptable range
        image_count += 1
    
    # Print the results for this image
    print(f'Image: {image_file}')
    print(f'Ground truth cells: {ground_truth_cells}')
    print(f'Detected cells: {detected_cells}')
    print(f'Within 5% margin: {abs(detected_cells - ground_truth_cells) <= error_margin}')
    
    # Optionally, you can display the image with detected cells
    plt.figure(figsize=(10, 10))
    plt.imshow(results[0].plot())  # This plots the bounding boxes on the image
    plt.title(f'Image: {image_file}')
    plt.axis('off')
    plt.show()

# Calculate the ACP across all images
if image_count > 0:
    ACP = (within_margin_count / image_count) * 100
    print(f"\nAccuracy per Cell Prediction (ACP): {ACP:.2f}%")
else:
    print("\nNo images processed. ACP cannot be calculated.")
