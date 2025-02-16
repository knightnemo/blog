import os
import re

def rename_files():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Print current directory for debugging
    print(f"Working directory: {current_dir}")
    
    # List all files before renaming
    print("Files before renaming:")
    for f in os.listdir(current_dir):
        print(f)
    
    # Updated pattern to match the actual file names
    pattern = r"扫描全能王 2025-02-15 21\.25_(\d+)\.jpg"
    
    # Get all files in the directory
    for filename in os.listdir(current_dir):
        match = re.match(pattern, filename)
        if match:
            # Get the number from the original filename
            number = match.group(1)
            # Create new filename
            new_name = f"{number}.jpg"
            
            # Full paths for old and new files
            old_path = os.path.join(current_dir, filename)
            new_path = os.path.join(current_dir, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")

if __name__ == "__main__":
    rename_files()
