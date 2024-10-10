# training_grids = [
#     (r'C:\Users\gsfny\Downloads\image.png', [0, 1, 2, 3, 4, 5]),
#     (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0000.png', [0, 1, 2, 3, 4, 5]),
#     (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0001.png', [0, 1, 2, 3, 4, 5]),
#     (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0002.png', [0, 1, 2, 3, 4, 5]),
#     (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0003.png', [0, 1, 2, 3, 4, 5]),
#     (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0004.png', [0, 1, 2, 3, 4, 5]),
#     (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0005.png', [0, 1, 2, 3, 4, 5]),
#     (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0006.png', [0, 1, 2, 3, 4, 5]),
# ]

import cv2
import numpy as np
import os
import argparse
import time
import shutil
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim

# Replace with actual paths to your images
training_grids = [
    (r'C:\Users\gsfny\Downloads\image.png', [0, 1, 2, 3, 4, 5]),
    (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0000.png', [0, 1, 2, 3, 4, 5]),
    (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0001.png', [0, 1, 2, 3, 4, 5]),
    (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0002.png', [0, 1, 2, 3, 4, 5]),
    (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0003.png', [0, 1, 2, 3, 4, 5]),
    (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0004.png', [0, 1, 2, 3, 4, 5]),
    (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0005.png', [0, 1, 2, 3, 4, 5]),
    (r'C:\Users\gsfny\Downloads\sd.webui\webui\outputs\txt2img-grids\2024-06-01\grid-0006.png', [0, 1, 2, 3, 4, 5]),
]

IMAGE_WIDTH = 512
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

def extract_features(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract HOG features
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def extract_images_from_grid(grid_image_path, grid_labels):
    print(f"Extracting images from grid: {grid_image_path}")
    grid_image = cv2.imread(grid_image_path)
    if grid_image is None:
        raise ValueError(f"Could not read image from path: {grid_image_path}")

    h, w, _ = grid_image.shape
    sub_image_height = h // 2
    sub_image_width = w // 3
    
    images = []
    labels = []
    for i in range(2):
        for j in range(3):
            sub_image = grid_image[i * sub_image_height:(i + 1) * sub_image_height, j * sub_image_width:(j + 1) * sub_image_width]
            sub_image = cv2.resize(sub_image, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize sub-image to consistent size
            images.append(sub_image)
            labels.append(grid_labels[i * 3 + j])
    
    print(f"Extracted {len(images)} images from grid.")
    return images, labels

def train_classifier():
    print("Training classifier...")
    X = []
    y = []
    for grid_image_path, grid_labels in training_grids:
        images, labels = extract_images_from_grid(grid_image_path, grid_labels)
        for image, label in zip(images, labels):
            features = extract_features(image)
            X.append(features)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Ensure test_size is appropriate for the number of samples and classes
    test_size = min(0.2, 6 / len(y))
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    print("Classifier trained.")
    
    # Evaluate the classifier using cross-validation
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")
    
    # Evaluate the classifier on the evaluation set
    y_pred = classifier.predict(X_eval)
    print("Evaluation results on evaluation set:")
    print(classification_report(y_eval, y_pred, zero_division=1))
    
    return classifier

def identify_group(image, classifier):
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Ensure consistent size
    features = extract_features(image)
    group = classifier.predict([features])[0]
    probabilities = classifier.predict_proba([features])[0]
    print(f"Image classified into group {group}")
    print(f"Classification probabilities: {probabilities}")
    return group

def compare_images(imageA, imageB):
    # Convert images to grayscale for SSIM comparison
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

def sort_images_within_groups(groups):
    print("Sorting images within groups...")
    sorted_groups = {}
    for group, images in groups.items():
        if not images:
            sorted_groups[group] = []
            continue
        print(f"Sorting group {group} with {len(images)} images.")
        # Pick a random image from the group
        random_image = images[0][0]
        # Calculate differences from the chosen image
        differences = [(compare_images(random_image, img), img, filename) for img, filename in images]
        # Sort images based on their difference
        differences.sort(key=lambda x: x[0])
        # Extract sorted images
        sorted_groups[group] = [(img, filename) for _, img, filename in differences]
    print("Images sorted within groups.")
    return sorted_groups

def copy_and_rename_images(folder, images, filenames, classifier, subdir):
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    print("Copying and renaming images...")
    timestamp = int(time.time())
    
    grouped_images = {i: [] for i in range(7)}
    
    for image, filename in zip(images, filenames):
        group = identify_group(image, classifier)
        grouped_images[group].append((image, filename))
    
    sorted_groups = sort_images_within_groups(grouped_images)
    
    counter = 1
    for group, images in sorted_groups.items():
        for image, filename in images:
            new_filename = f"group{group}_{str(counter).zfill(3)}_group{group}_simp_{timestamp}.png"
            old_filepath = os.path.join(folder, filename)
            new_filepath = os.path.join(subdir, new_filename)
            shutil.copy(old_filepath, new_filepath)
            print(f"Copied and renamed {filename} to {new_filename}")
            counter += 1

def main(folder):
    images, filenames = load_images_from_folder(folder)
    classifier = train_classifier()
    subdir = os.path.join(folder, f"renamed_images_{int(time.time())}")
    copy_and_rename_images(folder, images, filenames, classifier, subdir)
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group, sort, copy, and rename images based on classification")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    args = parser.parse_args()
    main(args.folder)
