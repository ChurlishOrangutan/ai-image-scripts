import cv2
import os
import argparse

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {frame_count}, FPS: {fps}")

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the current frame as a PNG file
    # if frame_number % 3 == 0:
        frame_filename = os.path.join(output_folder, f"frame_{str(frame_number).zfill(6)}.png")
        img = cv2.resize(frame, (615, 540))
        cv2.imwrite(frame_filename, img)
        print(f"Extracted frame {frame_number}/{frame_count}")

        frame_number += 1

    cap.release()
    print(f"Finished extracting frames to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an MP4 video and save them as PNG files.")
    parser.add_argument("video_path", type=str, help="Path to the input MP4 video file")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for PNG frames")
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_folder)
