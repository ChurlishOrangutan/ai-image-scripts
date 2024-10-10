import time
import cv2
import numpy as np
import os
import argparse
from skimage.metrics import structural_similarity as ssim

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

def load_images_from_folder(folder):
    print("Loading images from folder...")
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                filenames.append(filename)
                images.append(img)
                filenames.append(filename)
    return images, filenames

def compare_images(imageA, imageB):
    try:
        score, _ = ssim(imageA, imageB, full=True)
    except Exception as e:
        print(f"Error comparing images: {e}")
        score = -1
    return score

def merge_sort(images, filenames):
    if len(images) <= 1:
        return images, filenames

    mid = len(images) // 2
    left_images, left_filenames = merge_sort(images[:mid], filenames[:mid])
    right_images, right_filenames = merge_sort(images[mid:], filenames[mid:])

    return merge(left_images, left_filenames, right_images, right_filenames)

def merge(left_images, left_filenames, right_images, right_filenames):
    sorted_images = []
    sorted_filenames = []

    while left_images and right_images:
        if compare_images(left_images[0], right_images[0]) > compare_images(right_images[0], left_images[0]):
            sorted_images.append(left_images.pop(0))
            sorted_filenames.append(left_filenames.pop(0))
        else:
            sorted_images.append(right_images.pop(0))
            sorted_filenames.append(right_filenames.pop(0))

    sorted_images.extend(left_images)
    sorted_filenames.extend(left_filenames)
    sorted_images.extend(right_images)
    sorted_filenames.extend(right_filenames)

    return sorted_images, sorted_filenames

def rename_files(folder, sorted_filenames):
    num_files = len(sorted_filenames)
    padding_length = len(str(num_files))
    
    for i, filename in enumerate(sorted_filenames):
        timestamp = int(time.time())
        new_filename = f"{str(i+1).zfill(padding_length+2)}_{timestamp}.png"
        old_filepath = os.path.join(folder, filename)
        new_filepath = os.path.join(folder, new_filename)
        try:
            os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")
        except Exception as e:
            print(f"Failed to rename {filename} to {new_filename}: {e}")

def main(folder):
    images, filenames = load_images_from_folder(folder)
    sorted_images, sorted_filenames = merge_sort(images, filenames)
    print("Sorted filenames by similarity:")
    for filename in sorted_filenames:
        print(filename)
    rename_files(folder, sorted_filenames)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort images by similarity and rename them sequentially")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    args = parser.parse_args()
    main(args.folder)
