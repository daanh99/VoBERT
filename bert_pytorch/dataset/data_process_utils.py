import math
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


def split_abnormal_test_validation(df_abnormal, test_size=0.5, random_state=12):
    df_test_abnormal = df_abnormal.sample(frac=test_size, random_state=random_state)  # shuffle
    # everything that is not sampled
    df_valid_abnormal = df_abnormal.drop(df_test_abnormal.index)
    df_test_abnormal = df_test_abnormal.reset_index(drop=True)
    df_valid_abnormal = df_valid_abnormal.reset_index(drop=True)

    return df_test_abnormal, df_valid_abnormal


def save_dataset_to_file(filename, df, features, output_dir, log_file, semantic_embeddings, semantic_parsing):
    df_clone = df.copy()

    if semantic_parsing:
        if semantic_embeddings:
            df_templates = pd.read_csv(os.path.join(output_dir, log_file + "_templates.csv"))
            # replace all event ids in the arrays 'EventId' with the EventTemplate
            template_dict = df_templates.set_index('EventId').to_dict('dict')['EventTemplate']
            df_clone['EventId'] = df['EventId'].apply(lambda x: [template_dict[i] for i in x])
    else:
        if semantic_embeddings:
            df_templates = pd.read_csv(os.path.join(output_dir, log_file + "_structured.csv"))

            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df)

            # replace all event ids in the arrays 'EventId' with the EventTemplate
            template_dict = df_templates.set_index('LineId').to_dict('dict')['Content']
            df_clone['EventId'] = df['LineId'].apply(lambda x: [template_dict[i] for i in x])

    with open(os.path.join(output_dir, filename), 'w') as f:
        for _, row in tqdm(df_clone.iterrows(), total=len(df_clone.index), desc="Writing to file: " + filename):
            for val in zip(*row[features]):
                f.write('!;!'.join([str(v) for v in val]) + '!|!')
            f.write('\n')


def generate_datasets_new(output_dir, df_train, df_test_normal, df_test_abnormal, log_file,
                          semantic_embeddings, columns_to_write, parse_semantic, steps=5):
    # Save original split first
    ratio_unseen_normal_seqs, _ = checkRatioUnseen(output_dir, df_train, df_test_normal, df_test_abnormal,
                                                   verbose=True, save_to_csv=True)
    save_dataset_to_file('test_normal_' + str(round(ratio_unseen_normal_seqs, 4)),
                         df_test_normal, columns_to_write, output_dir, log_file,
                         semantic_embeddings, parse_semantic)
    save_dataset_to_file('train_' + str(round(ratio_unseen_normal_seqs, 4)),
                         df_train,
                         columns_to_write, output_dir, log_file, semantic_embeddings, parse_semantic)

    # add df_train and df_test_normal together. This is the new train set. Remove the sequences that contain the elements that appear in the least amount of sequences
    initial_train_size = len(df_train)
    # checkRatioUnseen(output_dir, df_train, df_test_normal, df_test_abnormal,
    #                  verbose=True, save_to_csv=True)
    df_all_samples = pd.concat([df_train, df_test_normal], ignore_index=True).reset_index(drop=True)
    train_vocab = uniqueValues(df_train)

    train_size_too_much = len(df_all_samples) - initial_train_size

    remove_indices_dict = []
    # Find in how many sequences of the train set each test_normal word occurs
    for train_word in tqdm(train_vocab, total=len(train_vocab),
                           desc="Finding in how many sequences each train word occurs"):
        remove_indices_current_word = df_all_samples[
            df_all_samples["EventId"].apply(lambda e: any([x == train_word for x in e]))].index.to_numpy()
        if len(remove_indices_current_word) > 0:
            remove_indices_dict.append(
                remove_indices_current_word)

    # Sort remove indices by amount of sequences that would have to be removed
    remove_indices_dict = sorted(remove_indices_dict, key=lambda x: len(x))

    remove_n_seqs = ((1 / steps) * train_size_too_much)

    while remove_n_seqs <= train_size_too_much:
        print("\n ------------------------------------\n")
        to_be_removed = list(unique_elements_in_sublists(remove_indices_dict, math.ceil(remove_n_seqs)))
        removed_rows = df_all_samples.iloc[to_be_removed]
        df_train_mod = df_all_samples.drop(to_be_removed).reset_index(drop=True)
        df_test_normal_mod = removed_rows.reset_index(drop=True)
        print("Removed {} sequences".format(len(to_be_removed)))

        # make train set smaller if needed
        if len(df_train_mod) > initial_train_size:
            print("Train set is larger than initial train set size")
            df_train_mod_correct_size = df_train_mod.sample(n=initial_train_size, random_state=42)
            not_sampled_into_train = df_train_mod.drop(df_train_mod_correct_size.index)
            df_test_normal_mod_correct_size = pd.concat([df_test_normal_mod, not_sampled_into_train]).reset_index(drop=True)

            # reset indices
            df_train_mod_correct_size = df_train_mod_correct_size.reset_index(drop=True)
            df_test_normal_mod_correct_size = df_test_normal_mod_correct_size.reset_index(drop=True)
            print("Reduced train set to {} sequences".format(len(df_train_mod)))
        elif len(df_train_mod) < initial_train_size:
            print("Train set is smaller than initial train set size. This should not happen!")
            exit(-1)
        else:
            print("Train set is of correct size")
            df_train_mod_correct_size = df_train_mod
            df_test_normal_mod_correct_size = df_test_normal_mod

        # save the sets
        ratio_unseen_normal_seqs, _ = checkRatioUnseen(output_dir, df_train_mod_correct_size, df_test_normal_mod_correct_size, df_test_abnormal,
                                                       verbose=True, save_to_csv=True)
        save_dataset_to_file('test_normal_' + str(round(ratio_unseen_normal_seqs, 4)),
                             df_test_normal_mod_correct_size, columns_to_write, output_dir, log_file,
                             semantic_embeddings, parse_semantic)
        save_dataset_to_file('train_' + str(round(ratio_unseen_normal_seqs, 4)),
                             df_train_mod_correct_size,
                             columns_to_write, output_dir, log_file, semantic_embeddings, parse_semantic)

        remove_n_seqs = remove_n_seqs + ((1 / steps) * train_size_too_much)


def unique_elements_in_sublists(x, N):
    if N == 0:
        return set()
    unique_elements = set()
    for sublist in x:
        for element in sublist:
            unique_elements.add(element)
            if len(unique_elements) >= N:
                return set(list(unique_elements)[:N])
    return unique_elements


# def generate_datasets_varying_unseen(output_dir, df_train, df_test_normal, df_test_abnormal, log_file,
#                                      semantic_embeddings=False):
#     ################
#     # Experiment Code to check if % of unseen logkeys
#     ################
#     min_train_size = int(len(df_train) / 4)
#     max_unseen_ratio = 1
#     add_removed_rows_to_test = False
#
#     # Unseen log keys mean log keys that are present in a test set (normal or abnormal), but not present in the train
#     # write header for the csv file
#     with open(os.path.join(output_dir, 'ratio_unseen.csv'), 'w') as f:
#         f.write(
#             'ratio_unseen_normal,ratio_unseen_abnormal,ratio_unseen_total,train_size,test_normal_size,test_abnormal_size,train_vocab,test_normal_vocab,test_abnormal_vocab,ratio_unseen_normal_sequences,ratio_unseen_abnormal_sequences,ratio_unseen_total_sequences\n')
#
#     ratio_unseen_normal_seqs, train_vocab_len = checkRatioUnseen(output_dir, df_train, df_test_normal, df_test_abnormal,
#                                                                  verbose=True, save_to_csv=True)
#
#     # save_dataset_to_file('test_normal_' + str(round(ratio_unseen_normal_seqs, 4)),
#     #                      df_test_normal, ["EventId", "element_labels"], output_dir, log_file, semantic_embeddings)
#     # save_dataset_to_file('train_' + str(round(ratio_unseen_normal_seqs, 4)), df_train,
#     #                      ["EventId", "element_labels"], output_dir, log_file, semantic_embeddings)
#
#     # Label 0 means normal, label 1 means abnormal
#     # Due to the training objective, only normal log keys are allowed to be present in the train set
#     # We want to find the ratio of log keys that are not present in the train set
#
#     # We can not lower the ratio of unseen log keys in the abnormal test set, because only normal log keys are allowed in the train set
#     # We can only increase the ratio of unseen log keys in the abnormal test set
#
#     # The original model has a bias for predicting unseen log keys as abnormal. The ratio of unseen log keys in the abnormal test set is much higher than the ratio of unseen log keys in the normal test set, making the prediction easy.
#     # We can make the prediction more fair by making the ratio of unseen log keys in the normal test set equal to the ratio of unseen log keys in the abnormal test set
#
#     # We do this by increasing the ratio of unseen keys for the normal test set
#     # We do this by removing sequences containing normal test log keys from the train set
#
#     # Increase ratio of unseen log keys in the normal test set:
#     # Remove sequences where the "EventId" sequence list contains a logkey that is also in the normal test set from the train set
#     # But don't remove all such sequences, remove them until the ratio of unseen log keys in the normal test set is equal to the ratio of unseen log keys in the abnormal test set
#
#     # test_normal_vocab = uniqueValues(df_test_normal)
#     # Remove sequences from the train set until the ratio of unseen log keys in the normal test set is equal to the ratio of unseen log keys in the abnormal test set
#     train_sets = [df_train.copy()]
#     test_normal_sets = [df_test_normal.copy()]
#     while ratio_unseen_normal_seqs < max_unseen_ratio and len(df_train) >= min_train_size:
#         test_normal_vocab = uniqueValues(df_test_normal)
#         remove_indices_dict = []
#         # Find in how many sequences of the train set each test_normal word occurs
#         for test_normal_word in tqdm(test_normal_vocab, total=len(test_normal_vocab),
#                                      desc="Finding in how many sequences each test_normal word occurs"):
#             # Find in how many sequences of the test_normal set this test_normal word occurs
#             word_ocurrs_in_n_seqs = len(
#                 df_test_normal[df_test_normal["EventId"].apply(lambda e: test_normal_word in e)])
#
#             remove_indices_current_word = df_train[
#                 df_train["EventId"].apply(lambda e: any([x == test_normal_word for x in e]))].index.to_numpy()
#             if len(remove_indices_current_word) > 0:
#                 remove_indices_dict.append(
#                     {"remove_indices": remove_indices_current_word, "n_seqs_normal": word_ocurrs_in_n_seqs})
#
#         # Sort remove indices by amount of sequences that would have to be removed
#         remove_indices_dict = sorted(remove_indices_dict, key=lambda x: (len(x["remove_indices"]), -x["n_seqs_normal"]))
#
#         amount_elements_to_remove_out_of_vocab = int(train_vocab_len / 2)
#         to_be_removed = [x['remove_indices'] for x in remove_indices_dict[:amount_elements_to_remove_out_of_vocab]]
#         # Flatten list and keep unique values
#         to_be_removed = list(set([item for sublist in to_be_removed for item in sublist]))
#
#         removed_rows = df_train.iloc[to_be_removed]
#         df_train = df_train.drop(to_be_removed).reset_index(drop=True)
#         if add_removed_rows_to_test:
#             df_test_normal = pd.concat([df_test_normal, removed_rows], ignore_index=True)
#         print("Removed {} sequences".format(len(to_be_removed)))
#
#         ratio_unseen_normal_seqs, train_vocab_len = checkRatioUnseen(output_dir, df_train, df_test_normal,
#                                                                      df_test_abnormal, verbose=True, save_to_csv=False)
#
#         if len(df_train) >= min_train_size:
#             train_sets.append(df_train)
#             test_normal_sets.append(df_test_normal)
#
#     print("Done removing sequences from train set, now shortening train sets to equal length")
#     # find the length of the shortest train set
#     min_train_set_length = min([len(x) for x in train_sets])
#
#     for i, df_train in enumerate(train_sets):
#         # reduce train set to the length of the shortest train set
#         df_train = df_train.sample(n=min_train_set_length, random_state=42)
#         ratio_unseen_normal_seqs, _ = checkRatioUnseen(output_dir, df_train, test_normal_sets[i], df_test_abnormal,
#                                                        verbose=True, save_to_csv=True)
#         save_dataset_to_file('test_normal_' + str(round(ratio_unseen_normal_seqs, 4)),
#                              test_normal_sets[i], ["EventId", "element_labels"], output_dir, log_file,
#                              semantic_embeddings)
#         save_dataset_to_file('train_' + str(round(ratio_unseen_normal_seqs, 4)),
#                              df_train,
#                              ["EventId", "element_labels"], output_dir, log_file, semantic_embeddings)


def get_sublists_for_n_unique(x, N):
    unique_elements = set()
    sublists = []
    for sublist in x:
        for element in sublist:
            unique_elements.add(element)
            if len(unique_elements) >= N:
                sublists.append(sublist)
                return sublists
        sublists.append(sublist)
    if len(unique_elements) < N:
        return x
    return sublists


def uniqueValues(df):
    # flatten list and find unique values
    return np.unique([item for sublist in df["EventId"] for item in sublist])


def checkRatioUnseen(output_dir, df_train, df_test_normal, df_test_abnormal, verbose=False, save_to_csv=True):
    train_vocab = uniqueValues(df_train)
    test_normal_vocab = uniqueValues(df_test_normal)
    test_abnormal_vocab = uniqueValues(df_test_abnormal)
    test_total_vocab = uniqueValues(pd.concat([df_test_normal, df_test_abnormal]))
    total_vocab = uniqueValues(pd.concat([df_train, df_test_normal, df_test_abnormal]))

    intersect_train_test_normal = np.intersect1d(train_vocab, test_normal_vocab)
    intersect_train_test_abnormal = np.intersect1d(train_vocab, test_abnormal_vocab)
    intersect_train_total = np.intersect1d(train_vocab, np.concatenate((test_normal_vocab, test_abnormal_vocab)))

    ratio_unseen_normal = 1 - (len(intersect_train_test_normal) / len(test_normal_vocab))
    ratio_unseen_abnormal = 1 - (len(intersect_train_test_abnormal) / len(test_abnormal_vocab))
    ratio_unseen_total = 1 - (len(intersect_train_total) / len(test_total_vocab))

    # Calculate ratio of sequences containing at least one unseen log key
    count_unseen_normal_sequences = len(
        df_test_normal[
            df_test_normal["EventId"].apply(lambda e: any([x not in intersect_train_test_normal for x in e]))])
    count_unseen_abnormal_sequences = len(
        df_test_abnormal[
            df_test_abnormal["EventId"].apply(lambda e: any([x not in intersect_train_test_abnormal for x in e]))])

    ratio_unseen_normal_sequences = count_unseen_normal_sequences / len(df_test_normal)
    ratio_unseen_abnormal_sequences = count_unseen_abnormal_sequences / len(df_test_abnormal)
    ratio_unseen_total_sequences = (count_unseen_normal_sequences + count_unseen_abnormal_sequences) / (
            len(df_test_normal) + len(df_test_abnormal))

    # Calculate average sequence length of all datasets
    df_total = pd.concat([df_train, df_test_normal, df_test_abnormal])
    average_sequence_length = np.mean([len(x) for x in df_total["EventId"]])

    if verbose:
        print()
        print("Train vocab size ", len(train_vocab), "Sequences: ", len(df_train))
        print("Test normal (non-attacks) vocab size: ", len(test_normal_vocab), ", ratio unseen : ",
              round(ratio_unseen_normal * 100, 2), "%", "Sequences: ", len(df_test_normal),
              ", ratio unseen sequences: ", round(ratio_unseen_normal_sequences * 100, 2))
        print("Test abnormal (attacks) vocab size: ", len(test_abnormal_vocab), ", ratio unseen : ",
              round(ratio_unseen_abnormal * 100, 2), "%", "Sequences: ", len(df_test_abnormal),
              ", ratio unseen sequences: ", round(ratio_unseen_abnormal_sequences * 100, 2))
        print("Total test vocab size: ", len(test_total_vocab), ", ratio unseen : ", round(ratio_unseen_total * 100, 2),
              "%", "Sequences: ", len(df_test_abnormal) + len(df_test_normal), ", ratio unseen sequences: ",
              round(ratio_unseen_total_sequences * 100, 2))
        print("Total vocab " + str(len(total_vocab)))
        print("Average sequence length: ", str(average_sequence_length))
        print()

    if save_to_csv:
        # Save numbers to csv
        with open(os.path.join(output_dir, 'ratio_unseen.csv'), 'a') as f:
            f.write(
                str(round(ratio_unseen_normal, 4))
                + "," + str(round(ratio_unseen_abnormal, 4))
                + "," + str(round(ratio_unseen_total, 4))
                + "," + str(len(df_train))
                + "," + str(len(df_test_normal))
                + "," + str(len(df_test_abnormal))
                + "," + str(len(train_vocab))
                + "," + str(len(test_normal_vocab))
                + "," + str(len(test_abnormal_vocab))
                + "," + str(len(total_vocab))
                + "," + str(ratio_unseen_normal_sequences)
                + "," + str(ratio_unseen_abnormal_sequences)
                + "," + str(ratio_unseen_total_sequences)
                + "," + str(average_sequence_length)
                + "\n")
    return ratio_unseen_normal_sequences, len(train_vocab)
