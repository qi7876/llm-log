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
            except ZeroDivisionError:
                print("Error calculating metrics: Division by zero.")
                results = {"precision": 0, "recall": 0, "f1_score": 0}
                return results

            results = {"precision": precision, "recall": recall, "f1_score": f1_score}

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
