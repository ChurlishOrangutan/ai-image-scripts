import time
import cv2
import numpy as np
import os
import argparse
from skimage.metrics import structural_similarity as ssim

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

def is_blue_to_white_sky(image):
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
    
    ul_mask = cv2.bitwise_or(ul_blue_mask, ul_white_mask)
    ur_mask = cv2.bitwise_or(ur_blue_mask, ur_white_mask)
    
    ul_ratio = cv2.countNonZero(ul_mask) / (100 * 100)
    ur_ratio = cv2.countNonZero(ur_mask) / (100 * 100)
    
    return ul_ratio > 0.5 and ur_ratio > 0.5

def filter_images_with_blue_to_white_sky(images, filenames):
    print("Filtering images with predominantly blue or white sky background...")
    filtered_images = []
    filtered_filenames = []
    for img, filename in zip(images, filenames):
        if is_blue_to_white_sky(img):
            filtered_images.append(img)
            filtered_filenames.append(filename)
    print(f"Filtered {len(filtered_images)} images with blue or white sky background.")
    return filtered_images, filtered_filenames

def compare_images(imageA, imageB):
    # Convert images to grayscale for SSIM comparison
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def basic_sort(images, filenames):
    print("Performing basic sort...")
    first_image = images[0]
    scores = [compare_images(first_image, img) for img in images]
    sorted_indices = np.argsort(scores)
    sorted_filenames = [filenames[i] for i in sorted_indices]
    sorted_images = [images[i] for i in sorted_indices]
    return sorted_images, sorted_filenames

def detailed_sort(images, filenames):
    print("Performing detailed sort...")
    n = len(images)
    p = 0
    scores = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            p =+ 1
            scores[i, j] = compare_images(images[i], images[j])
            scores[j, i] = scores[i, j]

    order = np.argsort(np.sum(scores, axis=0))
    sorted_filenames = [filenames[i] for i in order]
    print(f"Performing detailed sort iterations{p}")

    return sorted_filenames

def rename_files(folder, sorted_filenames):
    print("Renaming files...")
    num_files = len(sorted_filenames)
    padding_length = len(str(num_files))

    
    for i, filename in enumerate(sorted_filenames):
        timestamp = int(time.time())

        new_filename = f"sky{str(i+1).zfill(padding_length)}_{timestamp}.png"
        old_filepath = os.path.join(folder, filename)
        new_filepath = os.path.join(folder, new_filename)
        os.rename(old_filepath, new_filepath)
        print(f"Renamed {filename} to {new_filename}")
    return [f"{str(i+1).zfill(padding_length)}.png" for i in range(num_files)]

def main(folder):
    images, filenames = load_images_from_folder(folder)
    filtered_images, filtered_filenames = filter_images_with_blue_to_white_sky(images, filenames)
    basic_sorted_images, basic_sorted_filenames = basic_sort(filtered_images, filtered_filenames)
    sorted_filenames = detailed_sort(basic_sorted_images, basic_sorted_filenames)
    print("Sorted filenames by similarity:")
    for filename in sorted_filenames:
        print(filename)
    rename_files(folder, sorted_filenames)
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort images by similarity in two passes and rename them sequentially")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    args = parser.parse_args()
    main(args.folder)
