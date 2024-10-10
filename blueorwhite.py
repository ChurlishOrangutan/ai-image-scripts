import cv2
import numpy as np
import os
import argparse
import imagehash
from PIL import Image

def load_images_from_folder(folder):
    print("Loading images from folder...")
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):  # Sort filenames initially
        if filename.lower().endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    print(f"Loaded {len(images)} images.")
    return images, filenames

def is_blue_or_white(image):
    # Check the upper left and upper right corners
    ul_corner = image[:100, :100]
    ur_corner = image[:100, -100:]
    
    # Convert to HSV color space
    ul_hsv = cv2.cvtColor(ul_corner, cv2.COLOR_BGR2HSV)
    ur_hsv = cv2.cvtColor(ur_corner, cv2.COLOR_BGR2HSV)
    
    # Define blue to white color range in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    
    # Check if a significant portion is blue or white
    ul_blue_mask = cv2.inRange(ul_hsv, lower_blue, upper_blue)
    ur_blue_mask = cv2.inRange(ur_hsv, lower_blue, upper_blue)
    ul_white_mask = cv2.inRange(ul_hsv, lower_white, upper_white)
    ur_white_mask = cv2.inRange(ur_hsv, lower_white, upper_white)
    
    ul_blue_ratio = cv2.countNonZero(ul_blue_mask) / (100 * 100)
    ur_blue_ratio = cv2.countNonZero(ur_blue_mask) / (100 * 100)
    ul_white_ratio = cv2.countNonZero(ul_white_mask) / (100 * 100)
    ur_white_ratio = cv2.countNonZero(ur_white_mask) / (100 * 100)
    
    if ul_blue_ratio > 0.5 and ur_blue_ratio > 0.5:
        return "blue"
    elif ul_white_ratio > 0.5 and ur_white_ratio > 0.5:
        return "white"
    else:
        return "unknown"

def rename_files_with_color_and_duplicates(folder, images, filenames):
    print("Renaming files with color prefixes and checking for duplicates...")
    previous_hash = None
    
    for i, (image, filename) in enumerate(zip(images, filenames)):
        color_prefix = is_blue_or_white(image)
        dupe_prefix = ""
        
        # Calculate image hash
        current_hash = imagehash.average_hash(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        
        if previous_hash is not None and current_hash == previous_hash:
            dupe_prefix = "dupe_"
        
        new_filename = f"{color_prefix}_{filename}_{dupe_prefix}"
        old_filepath = os.path.join(folder, filename)
        new_filepath = os.path.join(folder, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed {filename} to {new_filename}")
        
        previous_hash = current_hash

def main(folder):
    images, filenames = load_images_from_folder(folder)
    rename_files_with_color_and_duplicates(folder, images, filenames)
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepend color and check for duplicates in images")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    args = parser.parse_args()
    main(args.folder)
