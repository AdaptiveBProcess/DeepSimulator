import pandas as pd
import readers.log_splitter as ls


def split_log(log, one_ts, size):
    splitter = ls.LogSplitter(log.data)
    train, validation = splitter.split_log('timeline_contained', size, one_ts)
    total_events = len(log.data)
    # Check size and change time splitting method if necessary
    if len(validation) < int(total_events * 0.1):
        train, validation = splitter.split_log('timeline_trace', size, one_ts)
    # Set splits
    validation = pd.DataFrame(validation)
    train = pd.DataFrame(train)
    return train, validation