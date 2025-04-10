import os
import shutil
import argparse

def is_folder_empty(folder_path):
    """
    Check if a folder is empty (has no files).
    A folder with only empty subfolders is still considered empty.
    """
    # Check each item in the folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # If it's a file, folder is not empty
        if os.path.isfile(item_path):
            return False
        
        # If it's a directory, check if it contains any files
        if os.path.isdir(item_path):
            if not is_folder_empty(item_path):
                return False
    
    # If we reach here, no files were found
    return True

def clean_empty_folders(parent_folder):
    """
    Check each subfolder in the parent folder.
    Keep folders with files, delete empty folders.
    """
    if not os.path.exists(parent_folder):
        print(f"Error: The folder '{parent_folder}' does not exist.")
        return
    
    # Get immediate subfolders
    subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found in '{parent_folder}'.")
        return
    
    # Track statistics
    kept_folders = 0
    deleted_folders = 0
    
    # Process each subfolder
    for subfolder in subfolders:
        if is_folder_empty(subfolder):
            print(f"Deleting empty folder: {subfolder}")
            shutil.rmtree(subfolder)
            deleted_folders += 1
        else:
            print(f"Keeping folder with files: {subfolder}")
            kept_folders += 1
    
    # Print summary
    print(f"\nSummary:")
    print(f"  - Kept {kept_folders} folder(s) with files")
    print(f"  - Deleted {deleted_folders} empty folder(s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete empty subfolders in a specified directory.")
    parser.add_argument("folder", help="The parent folder to check")
    parser.add_argument("--no-confirm", action="store_true", 
                       help="Skip confirmation prompt before deleting folders")
    
    args = parser.parse_args()
    
    if not args.no_confirm:
        confirmation = input(f"This will delete all empty subfolders in '{args.folder}'. Continue? (y/n): ")
        if confirmation.lower() != 'y':
            print("Operation cancelled.")
            exit()
    
    clean_empty_folders(args.folder)