import os
import shutil
import cv2
import numpy as np
import dlib
import re
import logging
import argparse
from skimage.metrics import structural_similarity as ssim

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\gsfny\\Downloads\\shape_predictor_68_face_landmarks.dat")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_face(image_path):
    """Extract face using dlib's facial landmark detection."""
    image = cv2.imread(image_path)
    
    faces = detector(image)

    if len(faces) > 0:
        # Select the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get facial landmarks
        shape = predictor(gray, face)
        
        # Extract face bounding box based on landmarks
        x_min = min([shape.part(i).x for i in range(68)])
        y_min = min([shape.part(i).y for i in range(68)])
        x_max = max([shape.part(i).x for i in range(68)])
        y_max = max([shape.part(i).y for i in range(68)])
        
        face_image = image[y_min:y_max, x_min:x_max]
        return face_image
    else:
        logging.warning(f"No face detected using dlib in {image_path}.")
        return None

def extract_face_again(image_path):
    """Extract face using OpenCV's Haar Cascade as a fallback."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = image[y:y+h, x:x+w]
        return face
    else:
        logging.warning(f"No face detected using Haar Cascade in {image_path}.")
        return None

def extract_image(image_path):
    """Return the entire grayscale image if no face is found."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def calculate_image_similarity(image1_path, image2_path):
    """Calculate similarity between two images, focusing on faces."""
    image1_face = extract_face(image1_path)
    image2_face = extract_face(image2_path)
    
    if image1_face is None or image2_face is None:
        image1_face = extract_face_again(image1_path)
        image2_face = extract_face_again(image2_path)
    
    if image1_face is None or image2_face is None:
        logging.info(f"Comparing full images instead of faces for {image1_path} and {image2_path}.")
        image1_face = extract_image(image1_path)
        image2_face = extract_image(image2_path)
    
    # Resize both images to a fixed size (adjust as needed)
    image1_resized = cv2.resize(image1_face, (720, 540))
    image2_resized = cv2.resize(image2_face, (720, 540))
    
    # Check if the image is already in grayscale (1 channel)
    if len(image1_resized.shape) == 3:  # If it has 3 channels (BGR)
        image1_gray = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
    else:
        image1_gray = image1_resized  # Already grayscale

    if len(image2_resized.shape) == 3:  # If it has 3 channels (BGR)
        image2_gray = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)
    else:
        image2_gray = image2_resized  # Already grayscale
    
    # Calculate Structural Similarity Index (SSIM)
    similarity, _ = ssim(image1_gray, image2_gray, full=True)
    
    return similarity


def find_most_similar_frame(current_frame_path, next_frame_set):
    highest_similarity = -1
    most_similar_frame = None
    
    for frame_path in next_frame_set:
        similarity = calculate_image_similarity(current_frame_path, frame_path)
        logging.info(f"Comparing {os.path.basename(current_frame_path)} to {os.path.basename(frame_path)}: similarity = {similarity}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_frame = frame_path

    logging.info(f"Selected {os.path.basename(most_similar_frame)} as the most similar frame.")
    return most_similar_frame

def get_batch_info(filename):
    # Extract job timestamp, batch size, and generation number using regex
    match = re.search(r'Job_(\d+)_Seed_\d+_BatchSz_(\d+)_GenNo_(\d+)_', filename)
    if match:
        job_timestamp = match.group(1)
        batch_size = int(match.group(2))
        gen_no = int(match.group(3))
        return job_timestamp, batch_size, gen_no
    return None, None, None

def identify_batches(frames):
    batches = []
    current_batch = []
    current_job_timestamp = None
    current_batch_size = 0

    for frame_path in frames:
        filename = os.path.basename(frame_path)
        job_timestamp, batch_size, gen_no = get_batch_info(filename)
        
        # If we encounter a new batch (GenNo == 1 or a different job timestamp), start a new batch
        if gen_no == 1 or job_timestamp != current_job_timestamp:
            if current_batch:
                batches.append(current_batch)
            current_batch = [frame_path]
            current_job_timestamp = job_timestamp
            current_batch_size = batch_size
        else:
            current_batch.append(frame_path)
        
        # If the current batch reaches the batch size, complete it
        if len(current_batch) == current_batch_size:
            batches.append(current_batch)
            current_batch = []

    # Add any remaining frames as the last batch
    if current_batch:
        batches.append(current_batch)

    return batches

def process_batches(input_folder, initial_frame_index, output_folder):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of all images sorted by name (assuming sequential naming)
    # frames = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('png', 'jpg', 'jpeg'))])
    frames = sorted(
        [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('png', 'jpg', 'jpeg'))],
        key=lambda x: os.path.getctime(os.path.join(input_folder, x))
    )
    # Identify batches
    batches = identify_batches(frames)
    
    # Set the initial frame path
    current_frame_path = frames[initial_frame_index]
    
    # Copy the initial frame to the output folder
    shutil.copy(current_frame_path, os.path.join(output_folder, os.path.basename(current_frame_path)))
    logging.info(f"Copied initial frame: {os.path.basename(current_frame_path)} to output folder.")
    
    # Process each identified batch, starting from the batch after the initial frame
    current_batch_index = (initial_frame_index // len(batches[0])) + 1
    while current_batch_index < len(batches):
        next_batch = batches[current_batch_index]
        logging.info(f"Processing batch starting with frame: {os.path.basename(next_batch[0])}")
        
        # Find the most similar frame in the next batch
        most_similar_frame = find_most_similar_frame(current_frame_path, next_batch)
        
        # Copy the most similar frame to the output folder
        if most_similar_frame:
            shutil.copy(most_similar_frame, os.path.join(output_folder, os.path.basename(most_similar_frame)))
            logging.info(f"Copied {os.path.basename(most_similar_frame)} to output folder.")
            
            # Update the current frame path to the newly selected frame
            current_frame_path = most_similar_frame
        
        # Move to the next batch
        current_batch_index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and copy the most similar frames within batches, focusing on faces using Dlib.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing frames.')
    parser.add_argument('initial_frame_index', type=int, help='Index of the initial frame to compare.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to store selected frames.')

    args = parser.parse_args()
    
    process_batches(args.input_folder, args.initial_frame_index, args.output_folder)
