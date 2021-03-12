# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:14:13 2020

@author: Manuel Camargo
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import copy
import shutil
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime


from sklearn.preprocessing import MaxAbsScaler
import keras.utils as ku
from nltk.util import ngrams
import utils.support as sup

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt


class DeepLearningGenerator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, ia_train, ia_valdn, parms):
        """constructor"""
        self.temp_output = os.path.join('output_files', sup.folder_id())
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
        self.ia_train = ia_train
        self.ia_valdn = ia_valdn
        self.parms = parms
        self.model_metadata = dict()
        self.is_safe = True
        self._load_model()
        
    # @safe_exec
    def _load_model(self) -> None:
        n_size = 5
        self.scaler = MaxAbsScaler()
        exec_times = dict()
        model_path = os.path.join(
            self.parms['ia_gen_path'],
            self.parms['file'].split('.')[0]+'_dl.h5')
        self.scaler.fit(self.ia_train[['inter_time']])
        if os.path.exists(model_path) and not self.parms['update_ia_gen']:
            self.model = load_model(model_path)
        elif os.path.exists(model_path) and self.parms['update_ia_gen']:
            self.model = load_model(model_path)
            self.is_safe = self._discover_model(True, n_size,
                log_time=exec_times, is_safe=self.is_safe)
        elif not os.path.exists(model_path):
            self.is_safe = self._discover_model(False, n_size,
                log_time=exec_times, is_safe=self.is_safe)
        # self._generate_traces(num_instances, start_time, n_size)

    # @safe_exec
    def _discover_model(self, compare, n_size, **kwargs):
        ia_valdn = copy.deepcopy(self.ia_valdn)
        # Transform features
        self.ia_train = self._transform_features(self.ia_train, self.scaler)
        ia_valdn = self._transform_features(ia_valdn, self.scaler)
        columns = self.ia_train.columns
        # vectorization
        serie_trn, y_serie_trn = self._vectorize(self.ia_train, columns, n_size) 
        serie_val, y_serie_val = self._vectorize(ia_valdn, columns, n_size)     
        # model training
        model = self._create_model(serie_trn.shape[1], 
                                        serie_trn.shape[2], 1)
        model = self._train_model(model,
                                       (serie_trn, y_serie_trn),
                                       (serie_val, y_serie_val))
        acc = model.evaluate(x=serie_val, y=y_serie_val, return_dict=True)
        metadata_file = os.path.join(
            self.parms['ia_gen_path'],
            self.parms['file'].split('.')[0]+'_dl_meta.json')
        # compare with existing model
        save = True
        if compare:
            # Loading of parameters from existing model
            if os.path.exists(metadata_file):
                with open(metadata_file) as file:
                    data = json.load(file)
                    # data = {k: v for k, v in data.items()}
                if data['loss'] < acc['loss']:
                    save = False
                    print('dont save')
        if save:
            self.model = model
            # best structure mining parameters
            self.model_metadata['loss'] = acc['loss']
            self.model_metadata['generated_at'] = (
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            # Copy best model to destination folder
            destintion = os.path.join(self.parms['ia_gen_path'],
                                      self.parms['file'].split('.')[0]+'_dl.h5')
            source = os.path.join(self.temp_output,
                                  self.parms['file'].split('.')[0]+'_dl.h5')

            shutil.copyfile(source, destintion)
            # Save metadata
            sup.create_json(self.model_metadata, metadata_file)
        # clean output folder
        shutil.rmtree(self.temp_output)
                    

    @staticmethod
    def _transform_features(split, scaler):
        split[['inter_time']] = scaler.transform(
            split[['inter_time']])
        split['daytime'] = np.divide(split['daytime'], 86400)
        day_dummies = ku.to_categorical(split.weekday, num_classes=7)
        day_dummies = pd.DataFrame(
            day_dummies, columns=['day_'+str(i) for i in range(7)])
        col_dummies = list(day_dummies.columns)
        split = pd.concat([split, day_dummies],
                                  axis=1, sort=False)
        columns = ['inter_time', 'daytime'] + col_dummies
        return split[columns]
        
        
    @staticmethod
    def _vectorize(split, columns, n_size):
        x_times_dict = dict()
        y_times_dict = dict()        
        for x in columns:
            serie = list(ngrams(split[x], n_size,
                                pad_left=True, left_pad_symbol=0))
            x_times_dict[x] = serie[:-1]
            if x == 'inter_time':
                y_serie = [x[-1] for x in serie]
                y_times_dict[x] = y_serie[1:]
            
        for key, value in x_times_dict.items():
            x_times_dict[key] = np.array(value)
            x_times_dict[key] = x_times_dict[key].reshape(
                (x_times_dict[key].shape[0], x_times_dict[key].shape[1], 1))
        serie = np.dstack(list(x_times_dict.values()))
        # Reshape y times attributes (suffixes, number of attributes)
        y_serie = np.array(list(y_times_dict.values()))[0]
        return serie, y_serie

    def _create_model(self, n_size, in_size, out_size):
        model = Sequential(name='generator')
        model.add(Input(shape=(n_size, in_size, ), name='inter_arrival'))
        model.add(GRU(100,
                      return_sequences=True,
                      dropout=0.2,
                      implementation=1))

        model.add(GRU(100,
                      return_sequences=False,
                      dropout=0.2,
                      implementation=1))

        model.add(Dense(out_size,
                        activation='linear',
                        name='t_output'))
        return model

    def _train_model(self, model, train_data, valdn_data):
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='mae', optimizer=opt)
        model.summary()

        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        # Output file
        output_path = os.path.join(self.temp_output,
                                   self.parms['file'].split('.')[0]+'_dl.h5')
        model_checkpoint = ModelCheckpoint(output_path,
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto')
        lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.5,
                                        patience=10,
                                        verbose=0,
                                        mode='auto',
                                        min_delta=0.0001,
                                        cooldown=0,
                                        min_lr=0)
        model.fit(train_data[0], train_data[1],
                  validation_data=valdn_data,
                  verbose=2,
                  callbacks=[early_stopping, lr_reducer, model_checkpoint],
                  epochs=100)
        return model

    # @safe_exec
    def generate(self, num_instances, start_time, n_size=5):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        tf.compat.v1.reset_default_graph()
        start_time = datetime.strptime(start_time, 
                                                "%Y-%m-%dT%H:%M:%S.%f+00:00")
        ia_time = 0.0
        daytime = start_time.time()
        daytime = daytime.second + daytime.minute*60 + daytime.hour*3600
        daytime = daytime / 86400
        day_dummies = ku.to_categorical(start_time.weekday(), num_classes=7)
        record = [ia_time, daytime] + list(day_dummies)
        x_t_ngram = np.zeros((1, n_size, len(record)))
        x_t_ngram = np.append(x_t_ngram, [[record]], axis=1)
        x_t_ngram = np.delete(x_t_ngram, 0, 1)

        timestamp = self.ia_valdn.timestamp.min()
        times = list()
        for i  in range(0, num_instances):
            preds = self.model.predict(x_t_ngram)
            preds[preds < 0] = 0
            ia_time = float(self.scaler.inverse_transform(preds)[0][0])
            timestamp += timedelta(seconds=ia_time*10)
            times.append({'caseid': 'Case'+str(i+1),
                          'timestamp': timestamp})
            time = timestamp.time()
            time = (time.second + time.minute*60 + time.hour*3600) / 86400
            day_dummies = ku.to_categorical(timestamp.weekday(), num_classes=7)
            record = [preds[0][0], time] + list(day_dummies)
            
            # Activities accuracy evaluation
            x_t_ngram = np.append(x_t_ngram, [[record]], axis=1)
            x_t_ngram = np.delete(x_t_ngram, 0, 1)
        self.times = pd.DataFrame(times)
        return self.times
    
    @staticmethod
    def _graph_timeline(log) -> None:
        time_series = log.copy()[['caseid', 'timestamp']]
        time_series['occ'] = 1
        time_series.set_index('timestamp', inplace=True)
        time_series.occ.rolling('3h').sum().plot(figsize=(30,10), linewidth=5, fontsize=10)
        plt.xlabel('Days', fontsize=20);
