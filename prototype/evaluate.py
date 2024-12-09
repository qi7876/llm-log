def calculateAccuracyAndRecall(groundTruthFile, predictionFile):
    try:
        with open(groundTruthFile, "r") as fGt, open(predictionFile, "r") as fPred:
            gtLines = fGt.readlines()
            predLines = fPred.readlines()

            # if len(gtLines) != len(predLines):
            #     print(
            #         "Error: Number of lines in ground truth and prediction files do not match."
            #     )
            #     return None

            truePositives = 0
            trueNegatives = 0
            falsePositives = 0
            falseNegatives = 0

            # for i in range(len(gtLines)):
            for i in range(602):
                gtLabel = gtLines[i].strip()
                predLabel = predLines[i].strip()

                # Ground Truth: Normal (-)
                if gtLabel.startswith("- "):
                    if "-" in predLabel:
                        trueNegatives += 1
                    elif "+" in predLabel:
                        falsePositives += 1
                    else:
                        print(
                            f"Error: Invalid prediction label at line {i+1}: {predLabel}"
                        )
                        return None

                # Ground Truth: Abnormal (no prefix)
                else:
                    if "+" in predLabel:
                        truePositives += 1
                    elif "-" in predLabel:
                        falseNegatives += 1
                    else:
                        print(
                            f"Error: Invalid prediction label at line {i+1}: {predLabel}"
                        )
                        return None

            accuracy = (
                (truePositives + trueNegatives) / len(gtLines)
                if len(gtLines) > 0
                else 0
            )
            recall = (
                truePositives / (truePositives + falseNegatives)
                if (truePositives + falseNegatives) > 0
                else 0
            )

            return {"accuracy": accuracy, "recall": recall}

    except FileNotFoundError:
        print("Error: One or both of the files were not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


groundTruthPath = "../dataset/BGL_2k.log"
predictionPath = "../test/output.txt"

results = calculateAccuracyAndRecall(groundTruthPath, predictionPath)

if results:
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
