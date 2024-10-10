import cv2
import numpy as np
import os
import argparse

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):  # Sort filenames initially
        if filename.lower().endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def calculate_orientation(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use Canny edge detector to detect edges
    edges = cv2.Canny(gray, 100, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0  # No contours found, assume straight
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle that bounds the contour
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[2]
    
    if angle < -45:
        angle = 90 + angle
    
    return angle

def adjust_orientation(image, angle):
    if angle < -10:
        # Leaning left, flip the image
        return cv2.flip(image, 1)
    return image

def process_images(folder):
    images, filenames = load_images_from_folder(folder)
    
    for img, filename in zip(images, filenames):
        angle = calculate_orientation(img)
        adjusted_image = adjust_orientation(img, angle)
        output_path = os.path.join(folder, filename)
        cv2.imwrite(output_path, adjusted_image)
        print(f"Processed {filename}: Angle = {angle:.2f}")

def main(folder):
    process_images(folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check and adjust model orientation in images")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    args = parser.parse_args()
    main(args.folder)
