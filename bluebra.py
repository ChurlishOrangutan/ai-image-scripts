import cv2
import os
import numpy as np
import argparse

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def is_darker_blue_bra_present(image):
    # Define the region of interest (ROI) for the bra area
    height, width, _ = image.shape
    roi = image[int(height * 0.4):int(height * 0.6), int(width * 0.3):int(width * 0.7)]  # Adjust as needed

    # Convert ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define the range for darker blue color in HSV
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 150])

    # Create a mask for blue color
    mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

    # Calculate the percentage of the ROI that is dark blue
    blue_ratio = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])

    # Return True if the ratio is above a certain threshold
    return blue_ratio > 0.2  # Adjust threshold as needed

def prepend_filenames_with_blue(folder):
    images, filenames = load_images_from_folder(folder)
    for image, filename in zip(images, filenames):
        if is_darker_blue_bra_present(image):
            new_filename = f"darkblue_{filename}"
            os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))
            print(f"Renamed {filename} to {new_filename}")

def main(folder):
    prepend_filenames_with_blue(folder)
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepend filenames if a darker blue bra is detected in the images")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    args = parser.parse_args()
    main(args.folder)
