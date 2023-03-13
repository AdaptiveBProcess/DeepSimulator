import pandas as pd


def split_log(log, one_ts, size):
    splitter = ls.LogSplitter(log.data)
    train, valdn = splitter.split_log('timeline_contained', size, one_ts)
    total_events = len(log.data)
    # Check size and change time splitting method if necesary
    if len(valdn) < int(total_events * 0.1):
        train, valdn = splitter.split_log('timeline_trace', size, one_ts)
    # Set splits
    valdn = pd.DataFrame(valdn)
    train = pd.DataFrame(train)
    return train, valdn