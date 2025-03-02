"""evaluate.py

This module is used to evaluate the accuracy and recall of the LogLLM.
@author: qi7876
"""

def calculateAccuracyAndRecall(ground_truth_file, prediction_file):
    try:
        with open(ground_truth_file, "r") as f_gt, open(prediction_file, "r") as f_pred:
            gt_lines = f_gt.readlines()
            pred_lines = f_pred.readlines()

            if len(gt_lines) != len(pred_lines):
                print(
                    "Error: Number of lines in ground truth and prediction files do not match."
                )
                return None

            true_positives = 0
            true_negatives = 0
            false_positives = 0
            false_negatives = 0

            for i in range(len(gt_lines)):
                # for i in range(1931):
                gt_label = gt_lines[i].strip()
                pred_label = pred_lines[i].strip()

                # Ground Truth: Normal (-)
                if gt_label.startswith("- "):
                    if "no" in pred_label:
                        true_negatives += 1
                    elif "yes" in pred_label:
                        false_positives += 1
                    else:
                        print(
                            f"Error: Invalid prediction label at line {i+1}: {pred_label}"
                        )
                        return None

                # Ground Truth: Abnormal (no prefix)
                else:
                    if "yes" in pred_label:
                        true_positives += 1
                    elif "no" in pred_label:
                        false_negatives += 1
                    else:
                        print(
                            f"Error: Invalid prediction label at line {i+1}: {pred_label}"
                        )
                        return None

            accuracy = (
                (true_positives + true_negatives) / len(gt_lines)
                if len(gt_lines) > 0
                else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives)
                if (true_positives + false_negatives) > 0
                else 0
            )

            return {"accuracy": accuracy, "recall": recall}

    except FileNotFoundError:
        print("Error: One or both of the files were not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


ground_truth_path = "../dataset/BGL/BGL_2k.log"
prediction_path = "output.txt"

results = calculateAccuracyAndRecall(ground_truth_path, prediction_path)

if results:
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
