import pathlib
import shutil

def empty_directory_pathlib(directory_path: str):
    """
    Deletes all files and subdirectories directly within the specified directory.
    The directory itself is NOT deleted.
    Args:
        directory_path: The path to the target directory to empty.
    """
    target_dir = pathlib.Path(directory_path)
    # 1. Basic Safety Checks
    if not target_dir.exists():
        print(f"Error: Directory '{directory_path}' does not exist.")
        return
    if not target_dir.is_dir():
        print(f"Error: Path '{directory_path}' is not a directory.")
        return
    # print(f"Scanning directory to empty: {target_dir}")
    deleted_files = 0
    deleted_dirs = 0
    error_count = 0
    # 2. Iterate through all items in the target directory
    for item in target_dir.iterdir():
        try:
            # 3. If it's a file or a symbolic link, delete it
            if item.is_file() or item.is_symlink():
                item.unlink()
                # print(f"  Deleted file/link: {item.name}")
                deleted_files += 1
            # 4. If it's a directory, delete it recursively
            elif item.is_dir():
                shutil.rmtree(item)
                # print(f"  Deleted directory and contents: {item.name}")
                deleted_dirs += 1
            # else: # Optional: Handle other types if necessary
            #     print(f"  Skipping unknown item type: {item.name}")
        except OSError as e:
            # 5. Handle potential errors during deletion
            print(f"  Error processing {item.name}: {e}")
            error_count += 1
    # print(f"\nFinished emptying {target_dir}.")
    # print(f"Successfully deleted {deleted_files} file(s)/link(s) and {deleted_dirs} director(y/ies).")
    if error_count > 0:
        print(f"Failed to process {error_count} item(s).")