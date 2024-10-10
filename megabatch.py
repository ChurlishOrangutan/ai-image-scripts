import os
import shutil
import cv2
import logging

def calculate_image_similarity(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    
    if image1 is None or image2 is None:
        raise ValueError(f"Error loading images. image1: {image1_path}, image2: {image2_path}")

    # Convert to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize images to the same size for comparison
    image1_gray = cv2.resize(image1_gray, (256, 256))
    image2_gray = cv2.resize(image2_gray, (256, 256))

    # Compute similarity using template matching
    score = cv2.matchTemplate(image1_gray, image2_gray, cv2.TM_CCOEFF_NORMED)[0][0]
    
    return score

def find_most_similar_frame(current_frame_path, next_batch):
    highest_similarity = -1
    most_similar_frame = None
    
    for frame_path in next_batch:
        similarity = calculate_image_similarity(current_frame_path, frame_path)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_frame = frame_path
    
    return most_similar_frame

def process_batches(input_folder, initial_frame_index, animation_folder, mega_batch_folder, mega_batch_size):
    # Ensure the output folders exist
    os.makedirs(animation_folder, exist_ok=True)
    os.makedirs(mega_batch_folder, exist_ok=True)

    # Enable logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get list of all images sorted by creation time
    frames = sorted(
        [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('png', 'jpg', 'jpeg'))],
        key=lambda x: os.path.getctime(os.path.join(input_folder, x))
    )
    
    # Set the initial frame path
    current_frame_path = frames[initial_frame_index]
    
    # Initialize the mega batch and frame counters
    mega_batch_count = 1
    animation_frame_count = 1  # This should count correctly for animation frames
    
    # Process frames in chunks of mega_batch_size
    for i in range(initial_frame_index, len(frames), mega_batch_size):
        next_batch = frames[i:i + mega_batch_size]
        
        if next_batch:
            logging.info(f"Processing mega batch #{mega_batch_count}")
            
            # Find the most similar frame from the next batch
            most_similar_frame = find_most_similar_frame(current_frame_path, next_batch)
            
            # Handle the animation frame (selected frame)
            if most_similar_frame:
                animation_filename = f"animation_frame_{animation_frame_count:03d}.png"
                animation_output_path = os.path.join(animation_folder, animation_filename)
                shutil.copy(most_similar_frame, animation_output_path)
                logging.info(f"Copied and renamed {os.path.basename(most_similar_frame)} to {animation_filename}")

                # Also store the entire mega batch in the mega batch folder
                for batch_frame in next_batch:
                    mega_batch_filename = f"animation_frame_{animation_frame_count:03d}_{mega_batch_count:03d}{os.path.basename(batch_frame)}"
                    mega_batch_output_path = os.path.join(mega_batch_folder, mega_batch_filename)
                    shutil.copy(batch_frame, mega_batch_output_path)
                    logging.info(f"Copied {os.path.basename(batch_frame)} to mega batch {mega_batch_count}")

                # Update the current frame to the newly selected one for the next comparison
                current_frame_path = most_similar_frame
            
            # Increment animation frame counter
            animation_frame_count += 1
        
        # Increment mega batch counter
        mega_batch_count += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process frames into separate animation and mega batch folders.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing frames.')
    parser.add_argument('initial_frame_index', type=int, help='Index of the initial frame to start with.')
    parser.add_argument('animation_folder', type=str, help='Path to the folder for selected animation frames.')
    parser.add_argument('mega_batch_folder', type=str, help='Path to the folder for mega batch frames.')
    parser.add_argument('mega_batch_size', type=int, help='Number of frames per mega batch.')

    args = parser.parse_args()
    
    process_batches(args.input_folder, args.initial_frame_index, args.animation_folder, args.mega_batch_folder, args.mega_batch_size)
