# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:13:15 2020

@author: Manuel Camargo
"""
import itertools
import numpy as np
# import random

from nltk.util import ngrams
import keras.utils as ku


class SequencesCreator():

    def __init__(self, one_timestamp, ac_index):
        """constructor"""
        self.one_timestamp = one_timestamp
        self.ac_index = ac_index
        

    def vectorize(self, model_type, log, params):
        vectorizer = self._get_vectorizer(model_type)
        return vectorizer(log, params)

    def _get_vectorizer(self, model_type):
        if model_type == 'basic':
            return self._vectorize_seq
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
        vec['pref']['weekday'] = ku.to_categorical(vec['pref']['weekday'], 
                                                   num_classes=7)
        # Transform prefixes in vectors
        vec['pref']['ac_index'] = np.array(vec['pref']['ac_index'])
        columns.remove('ac_index')
        columns.remove('weekday')
        for value in columns:
            vec['pref'][value] = np.array(vec['pref'][value])
            vec['pref'][value] = vec['pref'][value].reshape(
                (vec['pref'][value].shape[0], vec['pref'][value].shape[1], 1))
        for value in ['processing_time', 'waiting_time']:
            vec['next'][value] = np.array(vec['next'][value])
        # Features array
        features = vec['pref']['weekday']
        vec['pref'].pop('weekday', None)
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