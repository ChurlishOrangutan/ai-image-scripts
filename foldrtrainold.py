import cv2
import numpy as np
import os
import argparse
import shutil
from multiprocessing import Pool, cpu_count
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from skimage.feature import hog

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 1024

def load_images_from_folder(folder):
    print(f"Loading images from folder: {folder}")
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                images.append(img)
                filenames.append(filename)
    print(f"Loaded {len(images)} images from folder: {folder}")
    return images, filenames

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    return features

def load_training_data(folder_label_pair):
    folder, label = folder_label_pair
    print(f"Loading training data from folder: {folder} with label: {label}")
    X = []
    y = []
    images, _ = load_images_from_folder(folder)
    for image in images:
        features = extract_features(image)
        X.append(features)
        y.append(label)
    print(f"Loaded training data from folder: {folder} with label: {label}")
    return np.array(X), np.array(y)

def train_classifier(X, y):
    print("Starting training classifier...")
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    print("Classifier training complete.")
    
    cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")
    
    y_pred = classifier.predict(X_eval)
    print("Evaluation results on evaluation set:")
    print(classification_report(y_eval, y_pred, zero_division=1))
    
    return classifier

def classify_and_move_images(folder, classifier, label_map):
    print(f"Classifying and moving images from folder: {folder}")
    images, filenames = load_images_from_folder(folder)
    for image, filename in zip(images, filenames):
        features = extract_features(image)
        label = classifier.predict([features])[0]
        dest_folder = label_map[label]
        os.makedirs(dest_folder, exist_ok=True)
        old_filepath = os.path.join(folder, filename)
        new_filepath = os.path.join(dest_folder, filename)
        shutil.move(old_filepath, new_filepath)
        print(f"Moved {filename} to {new_filepath}")
    print(f"Classification and moving complete for folder: {folder}")

def sort_and_rename_files_in_directory(directory):
    print(f"Sorting and renaming files in directory: {directory}")
    files = sorted(os.listdir(directory))
    for index, filename in enumerate(files):
        if filename.lower().endswith('.png'):
            new_filename = f"{os.path.basename(directory)}_{str(index+1).zfill(3)}.png"
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f"Renamed {filename} to {new_filename}")
    print(f"Sorting and renaming complete for directory: {directory}")

def main(main_folder, training_dirs):
    print("Loading training data...")
    with Pool(cpu_count()) as pool:
        results = pool.map(load_training_data, [(folder, i) for i, folder in enumerate(training_dirs)])
    
    X = np.concatenate([res[0] for res in results])
    y = np.concatenate([res[1] for res in results])
    label_map = {i: folder for i, folder in enumerate(training_dirs)}

    print("Training classifier...")
    classifier = train_classifier(X, y)
    
    print("Classifying and moving images...")
    classify_and_move_images(main_folder, classifier, label_map)
    
    print("Sorting and renaming files in directories...")
    for directory in training_dirs:
        sort_and_rename_files_in_directory(directory)
    
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on folders and classify images into those folders")
    parser.add_argument("main_folder", type=str, help="Path to the main folder containing the images to classify")
    parser.add_argument("training_dirs", type=str, nargs='+', help="List of paths to training folders")
    args = parser.parse_args()
    main(args.main_folder, args.training_dirs)
