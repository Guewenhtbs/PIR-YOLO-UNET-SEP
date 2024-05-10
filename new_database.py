import os
import shutil

# Define the source and destination directories
src_dir = "Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information"
dst_dir = "new_database"

# Create the destination directory if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Loop through all subdirectories in the source directory
for dirpath, dirnames, filenames in os.walk(src_dir):
    # Loop through all files in the current subdirectory
    for filename in filenames:
        # Check if the filename contains "flair" or "lesionsegflair"
        if "Flair.nii" in filename or "LesionSeg-Flair.nii" in filename:
            # Construct the full file paths
            src_file = os.path.join(dirpath, filename)
            dst_file = os.path.join(dst_dir, filename)
            # Copy the file to the destination directory
            shutil.copy(src_file, dst_file)