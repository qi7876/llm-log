
import pandas as pd
import re
SYSTEM = 'Thunderbird'
LOG_COMPONENTS = ['Timestamp','Date','User','Month','Day','Time','Location','Component','PID','Content']
if __name__ == "__main__":
    csv = pd.read_csv(f'loghub/{SYSTEM}/{SYSTEM}_2k.log_structured.csv')
    df = pd.DataFrame(csv)
    length = len(df)
    label_list = ['']*length
    raw_log_list = ['']*length
    for idx, row in df.iterrows():
        label = row['EventId']
        raw_log = ''
        for component in LOG_COMPONENTS:
            raw_log += (str(row[component])+' ')
        label_list[idx] = label
        raw_log_list[idx] = raw_log
    output_df = pd.DataFrame({'raw_log': raw_log_list, 'label': label_list})
    output_df.to_csv(f"train_siamese_network_{SYSTEM}.csv", index=False, sep=',')
