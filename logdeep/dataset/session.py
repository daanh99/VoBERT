#see https://pinjiahe.github.io/papers/ISSRE16.pdf
import os
import pandas as pd

def session_window():
    pass


def sliding_window(raw_data, para):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=["timestamp", "Label", "EventId", "deltaT", "LineId"]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe columns=[eventids, time durations, label]
    """
    log_size = raw_data.shape[0]
    label_data = raw_data['Label']
    time_data = raw_data['timestamp']
    logkey_data = raw_data['EventId']
    deltaT_data = raw_data['deltaT']
    LineId_data = raw_data['LineId']

    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para['step_size']
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
            print("process {} time window".format(num_session), end='\r')

    for (start_index, end_index) in start_end_index_pair:
        dt = deltaT_data[start_index: end_index].values
        dt[0] = 0
        new_data.append([
            time_data[start_index: end_index].values,
            max(label_data[start_index:end_index]),
            logkey_data[start_index: end_index].values,
            dt,
            label_data[start_index: end_index].values,
            LineId_data[start_index: end_index].values
        ])

    assert len(start_end_index_pair) == len(new_data)

    print("min_len: ", para['min_len'])
    print("max_len: ", para['max_len'])
    print()
    print('There are %d sequences (sliding windows) in this dataset before removing too short sequences' % len(new_data))
    if len(new_data) > 0: print('The average length of a sequence (sliding window) is %d\n' % (sum([len(x[2]) for x in new_data]) / len(new_data)))

    new_data = [x for x in new_data if len(x[2]) >= para['min_len']]
    print('There are %d sequences (sliding windows) in this dataset after removing too short sequences' % len(new_data))
    if len(new_data) > 0: print('The average length of a sequence (sliding window) is %d \n' % (sum([len(x[2]) for x in new_data]) / len(new_data)))

    # remove too long sequences
    new_data = [
        tuple(
            sublist[:para['max_len']] if is_iterable(sublist) else sublist
            for sublist in x
        )
        if len(x[2]) > para['max_len']
        else x
        for x in new_data
    ]
    print('There are %d sequences (sliding windows) in this dataset after removing too short and too long sequences' % len(new_data))
    if len(new_data) > 0: print('The average length of a sequence (sliding window) is %d' % (sum([len(x[2]) for x in new_data]) / len(new_data)))
    print("-"*20)
    return pd.DataFrame(new_data, columns=["timestamp", "Label", "EventId", "deltaT", 'element_labels', 'LineId'])


def is_iterable(obj):
    """Check if an object is iterable"""
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def fixed_window(df, features, index, label, window_size='T'):
    """
    :param df: structured data after parsing
    features: datetime, eventid
    label: 1 anomaly/alert, 0 not anomaly
    :param window_size: offset datetime https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    :return:
    """
    df = df[features + [label]]
    agg_dict = {label: 'max'}
    for f in features:
        agg_dict[f] = _custom_resampler

    seq_df = df.set_index(index).resample(window_size).agg(agg_dict).reset_index()
    return seq_df


def _custom_resampler(array_like):
    return list(array_like)


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')

