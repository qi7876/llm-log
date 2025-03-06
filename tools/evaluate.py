import os
import sys
import getopt

# Change the working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""evaluate.py

This module is used to evaluate the accuracy and recall of the LogLLM.
prediction_file format:
{number}{response}
The number of lines should be in order.
For example:
1yse
2no
3no

Error:
1yes
2no
4no

@author: qi7876
"""

def validateFiles(ground_truth_file, prediction_file):
    """Validate if files exist and are readable."""
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return False
        
    if not os.path.exists(prediction_file):
        print(f"Error: Prediction file not found: {prediction_file}")
        return False
        
    if not os.access(ground_truth_file, os.R_OK):
        print(f"Error: Cannot read ground truth file: {ground_truth_file}")
        return False
        
    if not os.access(prediction_file, os.R_OK):
        print(f"Error: Cannot read prediction file: {prediction_file}")
        return False
        
    return True

def calculateAccuracyAndRecall(ground_truth_file, prediction_file):
    if not validateFiles(ground_truth_file, prediction_file):
        return None
        
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

            for i in range(len(pred_lines)):
                gt_label = gt_lines[i].strip()
                pred_label = pred_lines[i].strip()

                # Validate line number sequence
                if not pred_label.startswith(str(counter)):
                    print(f"ERROR: Prediction line {i+1} out of order: Expected {counter}, got {pred_label}")
                else:
                    counter += 1

                # Enhanced validation for prediction format
                if not ("yes" in pred_label.lower() or "no" in pred_label.lower()):
                    invalid_lines.append((i + 1, pred_label))
                    continue

                # Ground Truth: Normal (-)
                if gt_label.startswith("- "):
                    if "no" in pred_label.lower():
                        true_negatives += 1
                    elif "yes" in pred_label.lower():
                        false_positives += 1
                    else:
                        invalid_lines.append((i + 1, pred_label))

                # Ground Truth: Abnormal (no prefix)
                else:
                    if "yes" in pred_label.lower():
                        true_positives += 1
                    elif "no" in pred_label.lower():
                        false_negatives += 1
                    else:
                        invalid_lines.append((i + 1, pred_label))

            if invalid_lines:
                print(f"Found {len(invalid_lines)} invalid lines in prediction file:")
                for line_num, content in invalid_lines[:5]:  # Show only first 5 errors to avoid overload
                    print(f"  Line {line_num}: {content}")
                if len(invalid_lines) > 5:
                    print(f"  ... and {len(invalid_lines) - 5} more.")
                if len(invalid_lines) == len(pred_lines):
                    print("Error: All prediction lines are invalid.")
                    return None

            # Avoid division by zero errors
            try:
                precision = (
                    (true_positives + true_negatives) / (len(gt_lines) - len(invalid_lines))
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
            except ZeroDivisionError:
                print("Error calculating metrics: Division by zero.")
                return None

            results = {"precision": precision, "recall": recall, "f1_score": f1_score}
            
            # Add detailed statistics
            results.update({
                "true_positives": true_positives,
                "true_negatives": true_negatives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "invalid_lines": len(invalid_lines),
            })

            return results

    except UnicodeDecodeError:
        print("Error: One of the files contains invalid text encoding.")
        return None
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except PermissionError as e:
        print(f"Error: Permission denied when accessing files - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "g:p:h", ["ground_truth=", "prediction=", "help"])
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        print("Usage: python evaluate.py -g <ground_truth_file> -p <prediction_file>")
        sys.exit(1)

    ground_truth_path = None
    prediction_path = None

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage: python evaluate.py -g <ground_truth_file> -p <prediction_file>")
            print("Options:")
            print("  -g, --ground_truth  Path to the ground truth file")
            print("  -p, --prediction    Path to the prediction file")
            print("  -h, --help          Show this help message and exit")
            sys.exit(0)
        elif opt in ("-g", "--ground_truth"):
            ground_truth_path = arg
        elif opt in ("-p", "--prediction"):
            prediction_path = arg

    if not ground_truth_path or not prediction_path:
        print("Error: Both ground truth and prediction files must be specified.")
        print("Usage: python evaluate.py -g <ground_truth_file> -p <prediction_file>")
        sys.exit(1)

    results = calculateAccuracyAndRecall(ground_truth_path, prediction_path)

    if results:
        print("\nResults:")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print("\nDetailed Statistics:")
        print(f"True Positives: {results['true_positives']}")
        print(f"True Negatives: {results['true_negatives']}")
        print(f"False Positives: {results['false_positives']}")
        print(f"False Negatives: {results['false_negatives']}")
        if results['invalid_lines'] > 0:
            print(f"Invalid Lines: {results['invalid_lines']}")
    else:
        print("Evaluation failed. Please check the errors above.")
        sys.exit(2)
