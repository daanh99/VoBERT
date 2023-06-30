import pandas as pd
from tabulate import tabulate

def csv_to_rows(file_path, dataset_name):
    df = pd.read_csv(file_path)
    df_min = df.loc[df['ratio_unseen_normal_sequences'].idxmin()]
    df_max = df.loc[df['ratio_unseen_normal_sequences'].idxmax()]

    # Define the rows based on min and max
    # Similar to the previous script, but this time we only generate the rows, not a full table
    min_rows = [
        [dataset_name, 'Min', 'Train', None, None, df_min['train_vocab'], df_min['train_size']],
        ['', '', 'Test Normal', df_min['ratio_unseen_normal']*100, df_min['ratio_unseen_normal_sequences']*100, df_min['test_normal_vocab'], df_min['test_normal_size']],
        ['', '', 'Test Anomalous', df_min['ratio_unseen_abnormal']*100, df_min['ratio_unseen_abnormal_sequences']*100, df_min['test_abnormal_vocab'], df_min['test_abnormal_size']],
        ['', '', 'Test Total', df_min['ratio_unseen_total']*100, df_min['ratio_unseen_total_sequences']*100, None, df_min['test_normal_size']+df_min['test_abnormal_size']]
    ]

    max_rows = [
        [dataset_name, 'Max', 'Train', None, None, df_max['train_vocab'], df_max['train_size']],
        ['', '', 'Test Normal', df_max['ratio_unseen_normal']*100, df_max['ratio_unseen_normal_sequences']*100, df_max['test_normal_vocab'], df_max['test_normal_size']],
        ['', '', 'Test Anomalous', df_max['ratio_unseen_abnormal']*100, df_max['ratio_unseen_abnormal_sequences']*100, df_max['test_abnormal_vocab'], df_max['test_abnormal_size']],
        ['', '', 'Test Total', df_max['ratio_unseen_total']*100, df_max['ratio_unseen_total_sequences']*100, None, df_max['test_normal_size']+df_max['test_abnormal_size']]
    ]

    return min_rows + max_rows

def datasets_to_latex(dataset_info):
    all_rows = []
    for file_path, dataset_name in dataset_info:
        rows = csv_to_rows(file_path, dataset_name)
        all_rows.extend(rows)
    full_table = tabulate(all_rows, tablefmt="latex", headers=["Dataset", "Instability", "Part", "% Vocab Unseen", "% Seqs Containing Unseen", "Vocab Size", "Size"], floatfmt=".2f")
    return full_table

# Use the function like this:
dataset_info = [
    ('../output/bgl/matrix/ratio_unseen.csv', 'BLG'),
    ('../output/tbird/matrix/ratio_unseen.csv', 'TBird'),
    ('../output/hdfs/matrix/ratio_unseen.csv', 'HDFS')
]

print(datasets_to_latex(dataset_info))