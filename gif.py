import argparse
from PIL import Image
import os

def extract_frames(gif_path, output_folder):
    # Extract the base name of the GIF file without extension
    gif_basename = os.path.splitext(os.path.basename(gif_path))[0]
    
    # Open the animated GIF
    with Image.open(gif_path) as gif:
        # Ensure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        frame_number = 0
        while True:
            try:
                # Seek to the frame number
                gif.seek(frame_number)
            except EOFError:
                # No more frames in the GIF
                break

            # Create the filename for the frame
            frame_filename = os.path.join(output_folder, f"{gif_basename}_frame_{frame_number:03d}.png")

            # Save the current frame as a PNG file
            gif.save(frame_filename)
            print(f"Extracted frame {frame_number} and saved as {frame_filename}")

            frame_number += 1

def main(gif_path, output_folder):
    extract_frames(gif_path, output_folder)
    print(f"Finished extracting frames to {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from an animated GIF and save them as individual images.")
    parser.add_argument("gif_path", type=str, help="Path to the input animated GIF file")
    parser.add_argument("output_folder", type=str, help="Path to the output folder for extracted frames")
    args = parser.parse_args()

    main(args.gif_path, args.output_folder)
