from dataclasses import dataclass
from typing import List

import pandas as pd
from readers.log_splitter import LogSplitter


@dataclass
class FileExtensions:
    BPMN: str = '.bpmn'
    H5: str = '.h5'
    XES: str = '.xes'
    CSV: str = '.csv'
    JSON: str = '.json'
    EMB: str = '.emb'


@dataclass
class LogAttributes:
    CASE_ID: str = 'caseid'
    ACTIVITY: str = 'task'
    START_TIME: str = 'start_timestamp'
    END_TIME: str = 'end_timestamp'
    RESOURCE: str = 'user'
    ROLE: str = 'role'
    TIMESTAMP: str = 'timestamp'


@dataclass
class SequencesGenerativeMethods:
    PROCESS_MODEL: str = 'stochastic_process_model'
    TEST: str = 'test'

    def get_methods(self) -> List[str]:
        return list(self.__dict__.values())


@dataclass
class InterArrivalGenerativeMethods:
    PDF: str = 'pdf'
    DL: str = 'dl'
    MULTI_PDF: str = 'mul_pdf'
    TEST: str = 'test'
    PROPHET: str = 'prophet'

    def get_methods(self) -> List[str]:
        return list(self.__dict__.values())


@dataclass
class W2VecConcatMethod:
    SINGLE_SENTENCE: str = 'single_sentence'
    FULL_SENTENCE: str = 'full_sentence'
    WEIGHTING: str = 'weighting'

    def get_methods(self) -> List[str]:
        return list(self.__dict__.values())


@dataclass
class SplitMinerVersion:
    SM_V1: str = 'sm1'
    SM_V2: str = 'sm2'
    SM_V3: str = 'sm3'

    def get_methods(self) -> List[str]:
        return list(self.__dict__.values())


@dataclass
class EmbeddingMethods:
    DOT_PROD: str = 'emb_dot_product'
    DOT_PROD_TIMES: str = 'emb_dot_product_times'
    W2VEC: str = 'emb_w2vec'
    DOT_PROD_ACT_WEIGHT: str = 'emb_dot_product_act_weighting'

    @classmethod
    def get_base_model(cls, method):
        if method in [cls.DOT_PROD, cls.DOT_PROD_TIMES, cls.DOT_PROD_ACT_WEIGHT]:
            return 'Dot product'
        elif method == cls.W2VEC:
            return 'Word2vec'

    @classmethod
    def get_input_and_times_method(cls, method, include_times, concat_method):
        if method == cls.DOT_PROD:
            return 'N/A', False
        elif method == cls.DOT_PROD_TIMES:
            return 'Times', True
        elif method == cls.W2VEC:
            return concat_method, include_times
        elif method == cls.DOT_PROD_ACT_WEIGHT:
            return 'Activity weighting', include_times

    @classmethod
    def get_metrics_file_path(cls, method, include_times, concat_method, file_name):
        name = file_name.split('.')[0]
        _, inc_times = cls.get_input_and_times_method(method, include_times, concat_method)
        times = 'times' if inc_times else 'no_times'
        if method in [cls.DOT_PROD, cls.DOT_PROD_TIMES]:
            return f"ac_DP_{times}_{name}.csv"
        elif method == cls.DOT_PROD_ACT_WEIGHT:
            return f"ac_DP_act_weighting_{times}_{name}.csv"
        elif method == cls.W2VEC:
            return f"ac_W2V_{concat_method}_{times}_{name}.csv"

    @classmethod
    def get_matrix_file_name(cls, method, include_times, concat_method, file_name):
        name = file_name.split('.')[0]
        if method == cls.DOT_PROD:
            return f"ac_DP_{name}{FileExtensions.EMB}"
        elif method == cls.W2VEC:
            return f"ac_W2V_{concat_method}_{name}{FileExtensions.EMB}"
        elif method == cls.DOT_PROD_TIMES:
            return f"ac_DP_times_{name}{FileExtensions.EMB}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and include_times:
            return f"ac_DP_act_weighting_times_{name}{FileExtensions.EMB}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and not include_times:
            return f"ac_DP_act_weighting_no_times_{name}{FileExtensions.EMB}"

    @classmethod
    def get_model_file_name(cls, method, include_times, file_name):
        name = file_name.split('.')[0]
        if method == cls.DOT_PROD:
            return f"ac_DP_{name}_emb{FileExtensions.H5}"
        elif method == cls.DOT_PROD_TIMES:
            return f"ac_DP_times_{name}_emb{FileExtensions.H5}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and include_times:
            return f"ac_DP_act_weighting_times_{name}_emb{FileExtensions.H5}"
        elif method == cls.DOT_PROD_ACT_WEIGHT and not include_times:
            return f"ac_DP_act_weighting_no_times_{name}_emb{FileExtensions.H5}"
        else:
            return None

    def get_types(self) -> List[str]:
        return list(self.__dict__.values())


def split_log(log, one_ts, size):
    splitter = LogSplitter(log.data)
    train, validation = splitter.split_log('timeline_contained', size, one_ts)
    total_events = len(log.data)
    # Check size and change time splitting method if necessary
    if len(validation) < int(total_events * 0.1):
        train, validation = splitter.split_log('timeline_trace', size, one_ts)
    # Set splits
    validation = pd.DataFrame(validation)
    train = pd.DataFrame(train)
    return train, validation


OUTPUT_FILES = 'output_files'
