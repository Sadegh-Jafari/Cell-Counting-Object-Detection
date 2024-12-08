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

# Initialize variables for MAE and RMSE calculation
total_absolute_error = 0
total_squared_error = 0
image_count = 0

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
    
    # Run YOLOv8 detection on the enhanced image with the specified confidence and max detections
    results = model(enhanced_img, conf=0.25, max_det=1000)
    
    # Count the number of detected cells by YOLOv8
    detected_cells = len(results[0].boxes)
    
    # Calculate absolute and squared errors
    absolute_error = abs(detected_cells - ground_truth_cells)
    squared_error = (detected_cells - ground_truth_cells) ** 2
    
    # Update total errors and image count
    total_absolute_error += absolute_error
    total_squared_error += squared_error
    image_count += 1
    
    # Print the results for this image
    print(f'Image: {image_file}')
    print(f'Ground truth cells: {ground_truth_cells}')
    print(f'Detected cells: {detected_cells}')
    print(f'Absolute Error: {absolute_error}')
    print(f'Squared Error: {squared_error}')
    
    # Optionally, you can display the image with detected cells
    plt.figure(figsize=(10, 10))
    plt.imshow(results[0].plot())  # This plots the bounding boxes on the image
    plt.title(f'Image: {image_file} - Ground Truth: {ground_truth_cells}, Detected: {detected_cells}')
    plt.axis('off')
    plt.show()

# Calculate MAE and RMSE
if image_count > 0:
    MAE = total_absolute_error / image_count
    RMSE = np.sqrt(total_squared_error / image_count)
    print(f"\nMean Absolute Error (MAE): {MAE:.2f}")
    print(f"Root Mean Squared Error (RMSE): {RMSE:.2f}")
else:
    print("\nNo images processed. MAE and RMSE cannot be calculated.")
