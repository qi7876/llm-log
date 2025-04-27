import pathlib
import shutil
import chromadb
import os
import uuid


def create_database(
        file_path: str,
        db_dir: str = "chroma_db",
        collection_name: str = "documents"
) -> str:
    """
    Create a vector database from a local file.

    Args:
        file_path: Path to the file to be ingested
        db_dir: Directory to store the database
        collection_name: Name of the collection
        backup: Whether to backup existing database

    Returns:
        Path to the created database
    """
    # Create client
    client = chromadb.PersistentClient(path=db_dir)

    # Create or get collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Get the existing collection: {collection_name}")
    except Exception:
        collection = client.create_collection(name=collection_name)
        print(f"Create new collection: {collection_name}")

    # Read the file content
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Unable to find file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split content into chunks (simple splitting by newlines for example)
    chunks = content.split("\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Generate IDs for documents
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Add documents to the collection
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[{"source": file_path} for _ in chunks]
    )

    print(f"Add {len(chunks)} document chunks to the collection.")
    return db_dir

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

def calculate_accuracy_and_recall(ground_truth_file, prediction_file, line_num_file):
    counter = 1
    try:
        with open(ground_truth_file, "r") as f_gt, open(prediction_file, "r") as f_pred:
            gt_lines = f_gt.readlines()
            pred_lines = f_pred.readlines()

            if len(gt_lines) == 0:
                print("Error: Ground truth file is empty.")
                return None

            if len(pred_lines) == 0:
                print("Error: Prediction file is empty.")
                return None

            if len(gt_lines) != len(pred_lines):
                print(
                    f"Warning: Number of lines mismatch - Ground truth: {len(gt_lines)}, Prediction: {len(pred_lines)}"
                )
                # Use the minimum length to avoid index errors
                min_lines = min(len(gt_lines), len(pred_lines))
                gt_lines = gt_lines[:min_lines]
                pred_lines = pred_lines[:min_lines]
                print(f"Processing only the first {min_lines} lines.")

            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0
            invalid_lines = []
            error_details = []

            for i in range(len(pred_lines)):
                gt_label = gt_lines[i].strip()
                pred_label = pred_lines[i].strip()

                # Validate line number sequence
                if not pred_label.startswith(str(counter)):
                    print(f"ERROR: Prediction line {i + 1} out of order: Expected {counter}, got {pred_label}")
                else:
                    counter += 1

                # Enhanced validation for prediction format
                if not ("yes" in pred_label.lower() or "no" in pred_label.lower()):
                    invalid_lines.append((i + 1, pred_label))
                    error_details.append((i + 1, gt_label, pred_label, "Invalid format"))
                    continue

                # Ground Truth: Normal (-)
                if gt_label.startswith("- "):
                    if "no" in pred_label.lower():
                        true_negatives += 1
                    elif "yes" in pred_label.lower():
                        false_positives += 1
                        error_details.append((i + 1, gt_label, pred_label, "False Positive"))
                    else:
                        invalid_lines.append((i + 1, pred_label))
                        error_details.append((i + 1, gt_label, pred_label, "Invalid format"))

                # Ground Truth: Abnormal (no prefix)
                else:
                    if "yes" in pred_label.lower():
                        true_positives += 1
                    elif "no" in pred_label.lower():
                        false_negatives += 1
                        error_details.append((i + 1, gt_label, pred_label, "False Negative"))
                    else:
                        invalid_lines.append((i + 1, pred_label))
                        error_details.append((i + 1, gt_label, pred_label, "Invalid format"))

            if invalid_lines:
                print(f"Found {len(invalid_lines)} invalid lines in prediction file:")
                for line_num, content in invalid_lines[:5]:  # Show only first 5 errors to avoid overload
                    print(f"  Line {line_num}: {content}")
                if len(invalid_lines) > 5:
                    print(f"  ... and {len(invalid_lines) - 5} more.")
                if len(invalid_lines) == len(pred_lines):
                    print("Error: All prediction lines are invalid.")
                    return None

            if error_details:
                print("\nDetailed Error Analysis:")
                print("Line\tGround Truth\t\tPrediction\tError Type")
                print("-" * 65)
                for line_num, gt, pred, error_type in error_details:
                    # Truncate long new_log lines for better display
                    gt_display = (gt[:15] + '...') if len(gt) > 15 else gt
                    pred_display = (pred[:15] + '...') if len(pred) > 15 else pred
                    print(f"{line_num}\t{gt_display}\t{pred_display}\t\t{error_type}")
                    with open(line_num_file, 'a') as file:
                        file.write(str(line_num) + "\n")
            # Avoid division by zero errors
            try:
                precision = (
                    true_positives / (true_positives + false_positives)
                    if (len(gt_lines) - len(invalid_lines)) > 0
                    else 0
                )
                recall = (
                    true_positives / (true_positives + false_negatives)
                    if (true_positives + false_negatives) > 0
                    else 0
                )
                f1_score = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )
                results = {"precision": precision, "recall": recall, "f1_score": f1_score}
            except ZeroDivisionError:
                print("Error calculating metrics: Division by zero.")
                results = {"precision": 0, "recall": 0, "f1_score": 0}

            # Add detailed statistics
            results.update({
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "invalid_lines": len(invalid_lines),
                "error_details": error_details
            })

            return results

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None