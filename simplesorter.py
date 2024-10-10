import cv2
import numpy as np
import os
import argparse
import random
from skimage.metrics import structural_similarity as ssim
import time

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

def load_images_from_folder(folder):
    print("Loading images from folder...")
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):  # Sort filenames initially
        if filename.lower().endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                # Resize image to consistent dimensions
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                images.append(img)
                filenames.append(filename)
    print(f"Loaded {len(images)} images.")
    return images, filenames

def compare_images(imageA, imageB):
    # Convert images to grayscale for SSIM comparison
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def basic_sort_by_diff(images, filenames):
    print("Performing basic sort by difference...")
    # Pick a random image
    random_index = random.randint(0, len(images) - 1)
    random_image = images[random_index]
    random_filename = filenames[random_index]
    
    # Calculate differences from the randomly chosen image
    differences = []
    for img, filename in zip(images, filenames):
        if img is not random_image:
            try: 
                diff = compare_images(random_image, img)
                differences.append((diff, img, filename))
            except:
                print(f"whoopsiefiles: {filename}")
    # Sort images based on their difference
    differences.sort(key=lambda x: x[0], reverse=True)
    
    # Extract sorted filenames
    sorted_filenames = [random_filename] + [filename for _, _, filename in differences]
    return sorted_filenames

def rename_files(folder, filenames):
    print("Renaming files numerically with timestamp...")
    timestamp = int(time.time())
    
    for i, filename in enumerate(filenames):
        new_filename = f"{str(i+1).zfill(5)}_simp_{timestamp}.png"
        old_filepath = os.path.join(folder, filename)
        new_filepath = os.path.join(folder, new_filename)
        try:
            os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")
        except:
             print(f"Renamed {filename} to {new_filename} FAILED !!!!!")

def main(folder):
    images, filenames = load_images_from_folder(folder)
    sorted_filenames = basic_sort_by_diff(images, filenames)
    rename_files(folder, sorted_filenames)
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort images by difference and rename them numerically with timestamp")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    args = parser.parse_args()
    main(args.folder)
