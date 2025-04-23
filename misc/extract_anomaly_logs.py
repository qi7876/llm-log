def get_anomaly_logs(dataset_path, new_dataset_path):
    with open(dataset_path, 'r') as f, open(new_dataset_path, 'w') as new_f:
        for line in f:
            if not line.startswith('-'):
                new_f.write(line)

    return 0


if __name__ == '__main__':
    dataset_path = "../dataset/Thunderbird/Thunderbird.log.sub.key_event"
    new_dataset_path = "./Thunderbird_anomaly_log.txt"

    get_anomaly_logs(dataset_path, new_dataset_path)