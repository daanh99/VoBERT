import gc
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from bert_pytorch.dataset.data_process_utils import save_dataset_to_file, split_abnormal_test_validation, \
    generate_datasets_new
from logdeep.dataset.session import sliding_window
from logparser import Spell, Drain

tqdm.pandas()
pd.options.mode.chained_assignment = None

data_dir = os.path.abspath("redacted")


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+',  # hexadecimal
        r'\d+.\d+.\d+.\d+',
        # r'/\w+( )$'
        r'\d+'
    ]
    keep_para = False
    print("Going to parse private data")
    if parser_type == "drain":
        # the hyperparameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex,
                                 keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex,
                                 keep_para=keep_para)
        parser.parse(log_file)


def run(options):
    print("Starting pre-processing private dataset... v3")

    log_file = options["log_file"]
    output_dir = options["output_dir"]

    ##########
    # Parser #
    #########

    parse_log(data_dir, output_dir, log_file, 'drain')

    gc.collect()

    #########
    # Count #
    #########
    # count_anomaly()

    ##################
    # Transformation #
    ##################
    # mins
    window_size = 1 * 60
    step_size = 0.1 * 60
    train_ratio = 0.8

    df = pd.read_csv(f'{options["output_dir"]}{options["log_file"]}_structured.csv')

    df_log = pd.read_csv(f'{options["output_dir"]}{options["log_file"]}_raw.csv')

    # data preprocess
    df['datetime'] = pd.to_datetime(df_log['Time'], format='%Y-%m-%d.%H:%M:%S')
    df["Label"] = df_log["Label"]
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)
    # convert time to UTC timestamp
    # df['deltaT'] = df['datetime'].apply(lambda t: (t - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'))

    # sampling with fixed window
    # features = ["EventId", "deltaT"]
    # target = "Label"
    # deeplog_df = deeplog_df_transfer(df, features, target, "datetime", window_size=args.w)
    # deeplog_df.dropna(subset=[target], inplace=True)

    # sampling with sliding window
    deeplog_df = sliding_window(df[["timestamp", "Label", "EventId", "deltaT", "LineId"]],
                                para={"window_size": int(window_size) * 60,
                                      "step_size": int(step_size) * 60,
                                      "min_len": options['min_len'],
                                      "max_len": options['max_len']})

    #########
    # Train #
    #########
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True)  # shuffle
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    df_train = df_normal[:train_len]
    # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])

    print("training size {}".format(train_len))

    ###############
    # Test Normal #
    ###############
    df_test_normal = df_normal[train_len:]
    print("test normal size {}".format(normal_len - train_len))

    # del df_normal
    # del train
    # del test_normal
    # gc.collect()

    #################
    # Test Abnormal #
    #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]

    df_test_abnormal, df_valid_abnormal = split_abnormal_test_validation(df_abnormal, test_size=0.5)

    save_dataset_to_file('test_abnormal', df_test_abnormal, ["EventId", "element_labels"],
                         output_dir, log_file, options["semantic_embeddings"], options['parse_semantic'])
    save_dataset_to_file('valid_abnormal', df_valid_abnormal, ["EventId", "element_labels"],
                         output_dir, log_file, options["semantic_embeddings"], options['parse_semantic'])

    print('test abnormal size {}'.format(len(df_test_abnormal)))
    print('*' * 40)

    ################
    # Experiment Code to generate datasets varying unseen ratio
    ################
    generate_datasets_new(output_dir, df_train, df_test_normal, df_test_abnormal, log_file,
                          options["semantic_embeddings"], ["EventId", "element_labels"], options['parse_semantic'], 5)
