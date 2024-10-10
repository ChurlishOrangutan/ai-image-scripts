import os
import hashlib
import sys

def calculate_hash(file_path):
    """Calculate the MD5 hash of the file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def remove_duplicates(folder_path):
    """Remove duplicate PNG files in the specified folder."""
    file_hashes = {}
    duplicates = []

    print(f"Scanning folder: {folder_path}")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")

                file_hash = calculate_hash(file_path)
                print(f"File hash: {file_hash}")

                if file_hash in file_hashes:
                    duplicates.append(file_path)
                    print(f"Duplicate found: {file_path}")
                else:
                    file_hashes[file_hash] = file_path
                    print(f"File added to hash list: {file_path}")

    print("Removing duplicate files...")

    for duplicate in duplicates:
        os.remove(duplicate)
        print(f"Removed duplicate file: {duplicate}")

    print(f"Total duplicates removed: {len(duplicates)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_duplicates.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]
    remove_duplicates(folder_path)
