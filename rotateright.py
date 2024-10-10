import os
import argparse
from PIL import Image

def rotate_images(folder):
    # List all files in the directory
    for filename in os.listdir(folder):
        if filename.lower().endswith('.png'):
            file_path = os.path.join(folder, filename)
            try:
                # Open an image file
                with Image.open(file_path) as img:
                    # Rotate the image 90 degrees to the right
                    rotated_img = img.rotate(-90, expand=True)
                    # Save the rotated image
                    rotated_img.save(file_path)
                    print(f"Rotated and saved {filename}")
            except Exception as e:
                print(f"Failed to rotate {filename}: {e}")

def main(folder):
    rotate_images(folder)
    print("Finished rotating images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rotate all PNG images in a directory 90 degrees to the right")
    parser.add_argument("folder", type=str, help="Path to the folder containing the PNG images")
    args = parser.parse_args()
    main(args.folder)
