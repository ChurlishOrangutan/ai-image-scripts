import cv2
import os
import argparse
import time

def create_video_from_images(folder, fps):
    image_files = []
    for root, dirs, files in os.walk(folder):
        files = sorted(files)  # Sort files in each directory
        # files = sorted(files, key=lambda x: os.path.getctime(os.path.join(root, x)))
        for file in files:
            if file.lower().endswith('.png'):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No images to process.")
        return

    # Get the size of the first image to set the size of the video
    first_image_path = os.path.join(folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Get the current Unix timestamp
    timestamp = int(time.time())
    output_file = f"{timestamp}.mp4"

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for filename in image_files:
        print(filename)
        image_path = os.path.join(folder, filename)
        img = cv2.imread(image_path)
        video.write(img)
        # video.write(img)
    for filename in image_files[::-1]:
        print(filename)
        image_path = os.path.join(folder, filename)
        img = cv2.imread(image_path)
        video.write(img)
    # for i in range(0,10):
    #     filename = image_files[-i]
    #     print(filename)
    #     image_path = os.path.join(folder, filename)
    #     img = cv2.imread(image_path)
    #     video.write(img)
    #     video.write(img)
       


    video.release()
    print(f"Video saved as {output_file}")

def main(folder, fps):
    create_video_from_images(folder, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from sorted images")
    parser.add_argument("folder", type=str, help="Path to the folder containing the images")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for the video")
    args = parser.parse_args()
    main(args.folder, args.fps)
