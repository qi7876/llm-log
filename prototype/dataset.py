def loadDataset(datasetPath, maxLogNum) -> list[str]:
    logList = []
    logCount = 0
    with open(datasetPath, "r") as file:
        for line in file:
            if logCount >= maxLogNum:
                break
            if line.startswith("- "):
                line = line[2:]
            logList.append(line)
            logCount += 1

    return logList
