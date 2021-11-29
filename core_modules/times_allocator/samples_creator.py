# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:13:15 2020

@author: Manuel Camargo
"""
import pandas as pd
import itertools
import numpy as np

from nltk.util import ngrams
from keras.utils import np_utils as ku


class SequencesCreator():

    def __init__(self, one_timestamp, ac_index):
        """constructor"""
        self.one_timestamp = one_timestamp
        self.ac_index = ac_index
        

    def vectorize(self, model_type, log, params):
        vectorizer = self._get_vectorizer(model_type)
        return vectorizer(log, params)

    def _get_vectorizer(self, model_type):
        if model_type in ['basic', 'inter']:
            return self._vectorize_seq
        elif model_type == 'dual_inter':
            return self._dual_vectorize_seq
        elif model_type == 'inter_nt':
            return self._vectorize_nt_seq
        else:
            raise ValueError('Unexistent vectorizer')

    def _vectorize_seq(self, log, params):
        """
        Dataframe vectorizer.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        vec = {'pref': dict(), 'next': dict()}
        columns = [x for x in log.columns if x != 'caseid']
        # log = self.reformat_events(log, columns, params['one_timestamp'])
        log = self.reformat_events(log, columns, self.one_timestamp)
        # n-gram definition
        for i, _ in enumerate(log):
            for x in columns:
                serie = list(ngrams(log[i][x], params['n_size'],
                                    pad_left=True, left_pad_symbol=0))
                if x in ['processing_time', 'waiting_time']:
                    y_serie = [x[-1] for x in serie]
                    vec['next'][x] = (vec['next'][x] + y_serie if i > 0 else y_serie)
                    serie.insert(0, tuple([0 for i in range(params['n_size'])]))
                    serie.pop(-1)
                vec['pref'][x] = (vec['pref'][x] + serie if i > 0 else serie)
        # Transform weekday into one-hot encoding
        vec['pref']['st_weekday'] = ku.to_categorical(vec['pref']['st_weekday'], 
                                                   num_classes=7)
        # Transform prefixes in vectors
        vec['pref']['ac_index'] = np.array(vec['pref']['ac_index'])
        columns.remove('ac_index')
        columns.remove('st_weekday')
        for value in columns:
            vec['pref'][value] = np.array(vec['pref'][value])
            vec['pref'][value] = vec['pref'][value].reshape(
                (vec['pref'][value].shape[0], vec['pref'][value].shape[1], 1))
        for value in ['processing_time', 'waiting_time']:
            vec['next'][value] = np.array(vec['next'][value])
        # Features array
        features = vec['pref']['st_weekday']
        vec['pref'].pop('st_weekday', None)
        for x in columns:
            features = np.concatenate((features, vec['pref'][x]), axis=2)
            vec['pref'].pop(x, None)
        vec['pref']['features'] = features
        # Output array
        vec['next']['processing_time'] = (
            vec['next']['processing_time']
            .reshape((vec['next']['processing_time'].shape[0], 1)))
        vec['next']['waiting_time'] = (
            vec['next']['waiting_time']
            .reshape((vec['next']['waiting_time'].shape[0], 1)))
        vec['next']['expected'] = np.concatenate(
            (vec['next']['processing_time'], 
             vec['next']['waiting_time']), axis=1)
        vec['next'].pop('processing_time', None)
        vec['next'].pop('waiting_time', None)
        return vec

    
    def _dual_vectorize_seq(self, log, params):
        """
        Dataframe vectorizer.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        ngram_size = params['n_size']
        vec = {'proc_model': dict(), 'waiting_model': dict()}
        
        def create_vector(df, exp_col, week_col):
            dt_prefixes = list()
            dt_expected = list()
            cols = list(df.columns)
            for key, group in df.groupby('caseid'):
                dt_prefix = pd.DataFrame(0, index=range(ngram_size), 
                                         columns=cols+['ngram_num'])
                dt_prefix['caseid'] = key
                dt_prefix = pd.concat([dt_prefix, group], axis=0)
                dt_prefix = dt_prefix.iloc[:-1]
                dt_expected.append(group[exp_col])
                for nr_events in range(0, len(group)):
                    tmp = dt_prefix.iloc[nr_events:nr_events+ngram_size].copy()
                    tmp['ngram_num'] = nr_events
                    dt_prefixes.append(tmp)
            dt_prefixes = pd.concat(dt_prefixes, axis=0, ignore_index=True)
            dt_expected = pd.concat(dt_expected, axis=0, ignore_index=True)
            # One-hot encode weekday
            ohe_df = ku.to_categorical(
                dt_prefixes[week_col].to_numpy().reshape(-1, 1), num_classes=7)
            # Create a Pandas DataFrame of the hot encoded column
            ohe_df = pd.DataFrame(ohe_df)
            ohe_df.rename(
                columns= {col: 'day_'+str(col) for col in ohe_df.columns},
                inplace=True)
            # Concat with original data
            dt_prefixes = pd.concat([dt_prefixes, ohe_df],
                                    axis=1).drop([week_col], axis=1)
            num_samples = len(
                dt_prefixes[['caseid', 'ngram_num']].drop_duplicates())
            dt_prefixes.drop(columns={'caseid', 'ngram_num'}, inplace=True)
            num_columns = len(dt_prefixes.columns)        
            dt_prefixes = dt_prefixes.to_numpy().reshape(num_samples,
                                                         ngram_size,
                                                         num_columns)
            dt_expected = dt_expected.to_numpy().reshape((num_samples, 1))
            return dt_prefixes, dt_expected
        
        caseid_col = ['caseid']
        st_cols = (caseid_col + ['ac_index', 'processing_time'] +
                   [c_n for c_n in log.columns if 'st_' in c_n])
        end_cols = (caseid_col + ['n_ac_index', 'waiting_time'] +
                    [c_n for c_n in log.columns if 'end_' in c_n])
        st_train, st_expected = create_vector(log[st_cols],
                                              'processing_time',
                                              'st_weekday')
        vec['proc_model']['pref'] = dict()
        vec['proc_model']['pref']['ac_index'] = st_train[:,:,0]
        vec['proc_model']['pref']['features'] = st_train[:,:,1:]
        vec['proc_model']['next'] = st_expected
        end_train, end_expected = create_vector(log[end_cols],
                                               'waiting_time',
                                               'end_weekday')
        vec['waiting_model']: dict()
        vec['waiting_model']['pref'] = dict()
        vec['waiting_model']['pref']['ac_index'] = end_train[:,:,0]
        vec['waiting_model']['pref']['features'] = end_train[:,:,1:]
        vec['waiting_model']['next'] = end_expected
        return vec

    def _vectorize_nt_seq(self, log, params):
        """
        Dataframe vectorizer.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        ngram_size = params['n_size']
        vec = {'proc_model': dict(), 'waiting_model': dict()}

        caseid_col = ['caseid']
        week_col = 'st_weekday'
        exp_col = ['processing_time', 'waiting_time'] 
        cols = (caseid_col + ['ac_index', 'n_ac_index'] + exp_col +
                [c_n for c_n in log.columns if 'st_' in c_n])
        log = log[cols]
        dt_prefixes = list()
        dt_expected = list()
        for key, group in log.groupby('caseid'):
            dt_prefix = pd.DataFrame(0, index=range(ngram_size), 
                                     columns=cols+['ngram_num'])
            dt_prefix['caseid'] = key
            dt_prefix = pd.concat([dt_prefix, group], axis=0)
            dt_prefix = dt_prefix.iloc[:-1]
            dt_expected.append(group[exp_col])
            for nr_events in range(0, len(group)):
                tmp = dt_prefix.iloc[nr_events:nr_events+ngram_size].copy()
                tmp['ngram_num'] = nr_events
                dt_prefixes.append(tmp)
        dt_prefixes = pd.concat(dt_prefixes, axis=0, ignore_index=True)
        dt_expected = pd.concat(dt_expected, axis=0, ignore_index=True)
        # One-hot encode weekday
        ohe_df = ku.to_categorical(
            dt_prefixes[week_col].to_numpy().reshape(-1, 1), num_classes=7)
        # Create a Pandas DataFrame of the hot encoded column
        ohe_df = pd.DataFrame(ohe_df)
        ohe_df.rename(
            columns= {col: 'day_'+str(col) for col in ohe_df.columns},
            inplace=True)
        # Concat with original data
        dt_prefixes = pd.concat([dt_prefixes, ohe_df],
                                axis=1).drop([week_col], axis=1)
        num_samples = len(
            dt_prefixes[['caseid', 'ngram_num']].drop_duplicates())
        dt_prefixes.drop(columns={'caseid', 'ngram_num'}, inplace=True)
        num_columns = len(dt_prefixes.columns)        
        dt_prefixes = dt_prefixes.to_numpy().reshape(num_samples,
                                                     ngram_size,
                                                     num_columns)
        dt_expected = dt_expected.to_numpy().reshape((num_samples, 2))
        
        vec['pref'] = dict()
        vec['pref']['ac_index'] = dt_prefixes[:,:,0]
        vec['pref']['n_ac_index'] = dt_prefixes[:,:,1]
        vec['pref']['features'] = dt_prefixes[:,:,2:]
        vec['next'] = dt_expected
        return vec

    # =============================================================================
    # Reformat events
    # =============================================================================
    def reformat_events(self, log, columns, one_timestamp):
        """Creates series of activities, roles and relative times per trace.
        parms:
            self.log: dataframe.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
        Returns:
            list: lists of activities, roles and relative times.
        """
        temp_data = list()
        log_df = log.to_dict('records')
        key = 'end_timestamp' if one_timestamp else 'start_timestamp'
        log_df = sorted(log_df, key=lambda x: (x['caseid'], key))
        for key, group in itertools.groupby(log_df, key=lambda x: x['caseid']):
            trace = list(group)
            temp_dict = dict()
            for x in columns:
                serie = [y[x] for y in trace]
                if x == 'waiting_time':
                    serie.pop(0)
                    serie.append(0)
                temp_dict = {**{x: serie}, **temp_dict}
            temp_dict = {**{'caseid': key}, **temp_dict}
            temp_data.append(temp_dict)
        return temp_data