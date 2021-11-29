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
from keras.utils import np_utils as ku
from nltk.util import ngrams
import utils.support as sup

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Dropout, Flatten
from tensorflow.keras.layers import Conv1DTranspose, AveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt
from pickle import dump, load
import warnings


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
        model_path = os.path.join(self.parms['ia_gen_path'],
                                  self.parms['file'].split('.')[0]+'_dl.h5')
        # Save path(s) if the model exists 
        self.model_path = model_path
        model_exist = os.path.exists(model_path)
        self.parms['model_path'] = model_path
        # Discover and compare
        if not model_exist or self.parms['update_ia_gen']:
            scaler = MaxAbsScaler()
            scaler.fit(self.ia_train[['inter_time']])
            acc = self._discover_model(scaler)
            save, metadata_file = self._compare_models(acc, 
                self.parms['update_ia_gen'], model_path)
            if save:
                self._save_model(metadata_file, acc)
            else:
                shutil.rmtree(self.temp_output)
            # Save basic features scaler
            name = self.model_path.replace('_dl.h5', '')
            dump(scaler, open(name+'_ia_scaler.pkl','wb'))


    def _compare_models(self, acc, model_exist, file):
        metadata_file = os.path.join(self.parms['ia_gen_path'],
            self.parms['file'].split('.')[0]+'_dl_meta.json')
        # compare with existing model
        save = True
        if model_exist:
            # Loading of parameters from existing model
            if os.path.exists(metadata_file):
                with open(metadata_file) as file:
                    data = json.load(file)
                if data['loss'] < acc['loss']:
                    save = False
        return save, metadata_file


    # @safe_exec
    def _discover_model(self, scaler):
        n_size = 5
        ia_valdn = copy.deepcopy(self.ia_valdn)
        # Transform features
        self.ia_train = self._transform_features(self.ia_train, scaler)
        ia_valdn = self._transform_features(ia_valdn, scaler)
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
        return model.evaluate(x=serie_val, y=y_serie_val, return_dict=True)
    
    
    def _save_model(self, metadata_file, acc):
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


    def _vectorize(self, log, cols, ngram_size):
        """
        Dataframe vectorizer.
        parms:
            columns: list of features to vectorize.
            parms (dict): parms for training the network
        Returns:
            dict: Dictionary that contains all the LSTM inputs.
        """
        vec = dict()

        week_col = 'weekday'
        exp_col = ['inter_time'] 
        log = log[cols]
        num_samples = len(log)
        dt_prefixes = list()
        dt_expected = list()
        # for key, group in log.groupby('caseid'):
        dt_prefix = pd.DataFrame(0, index=range(ngram_size), columns=cols)
        dt_prefix = pd.concat([dt_prefix, log], axis=0)
        dt_prefix = dt_prefix.iloc[:-1]

        dt_expected.append(log[exp_col])
        for nr_events in range(0, num_samples):
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
        dt_prefixes.drop(columns={'ngram_num'}, inplace=True)
        num_columns = len(dt_prefixes.columns)        
        dt_prefixes = dt_prefixes.to_numpy().reshape(num_samples,
                                                     ngram_size,
                                                     num_columns)
        dt_expected = dt_expected.to_numpy().reshape((num_samples, 1))
        
        return dt_prefixes, dt_expected
    
    @staticmethod
    def _transform_features(split, scaler):
        split[['inter_time']] = scaler.transform(
            split[['inter_time']])
        split['daytime'] = np.divide(split['daytime'], 86400)
        columns = ['inter_time', 'daytime', 'weekday']
        return split[columns]

        
    def _create_model(self, n_size, in_size, out_size):
        series_input = Input(shape=(n_size, in_size, ), name='inter_arrival')
        gru_1 = GRU(100, return_sequences=True, dropout=0.2,
                      implementation=1)(series_input)
        gru_2 = GRU(100, return_sequences=False, dropout=0.2,
                      implementation=1)(gru_1)
        output = Dense(out_size, activation='linear', name='t_output')(gru_2)
        model = Model(inputs=[series_input], outputs=[output])
        opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        model.compile(loss='mae', optimizer=opt)
        model.summary()
        return model

    # def _create_model(self, n_size, in_size, out_size):
    #     series_input = Input(shape=(n_size, in_size, ), name='inter_arrival')
    #     # gru_1 = GRU(100, return_sequences=True, dropout=0.2,
    #     #               implementation=1)(series_input)
    #     gru_2 = GRU(100, return_sequences=False, dropout=0.2,
    #                   implementation=1)(series_input)
    #     output = Dense(out_size, activation='linear', name='t_output')(gru_2)
    #     model = Model(inputs=[series_input], outputs=[output])
    #     opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #     model.compile(loss='mae', optimizer=opt)
    #     model.summary()

    # def _create_model(self, n_size, in_size, out_size):
    #     series_input = Input(shape=(n_size, in_size, ), name='inter_arrival')
    #     conv_1 = Conv1DTranspose(filters = 32, 
    #                      data_format = 'channels_last',
    #                      kernel_size=2, 
    #                      strides=1,   
    #                      activation='relu')(series_input)
    #     #model.add(MaxPooling2D(pool_size=(2, 1)))
    #     pool = AveragePooling1D(pool_size=2)(conv_1)
    #     drop_1 = Dropout(0.2)(pool)
    #     conv_2 = Conv1DTranspose(filters = 12, 
    #                      data_format = 'channels_last',
    #                      kernel_size=2, 
    #                      strides=1,   
    #                      activation='relu')((drop_1))
    #     #model.add(MaxPooling2D(pool_size=(2, 1)))
    #     drop_2 = Dropout(0.1)(conv_2)
    #     flat = Flatten()(drop_2)
    #     drop_3 = Dropout(0.2)(flat)
    #     dense = Dense(45, activation='relu')(drop_3)
    #     output = Dense(1)(dense)
    #     model = Model(inputs=[series_input], outputs=[output])
    #     opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    #     model.compile(loss='mae', optimizer=opt)
    #     model.summary()
    #     return model


    def _train_model(self, model, train_data, valdn_data):

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
                  epochs=self.parms['epochs'])
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
        g, session, model = self._load_models(self.model_path)
        
        name = self.model_path.replace('_dl.h5', '')
        scaler = load(open(name+'_ia_scaler.pkl', 'rb'))
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

        timestamp = start_time
        times = list()
        for i  in range(0, num_instances):
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                with g.as_default():
                    with session.as_default():
                        preds = model.predict(x_t_ngram)
            preds[preds < 0] = 0.000001
            ia_time = float(scaler.inverse_transform(preds)[0][0])
            timestamp += timedelta(seconds=int(round(ia_time)))
            # timestamp += timedelta(seconds=int(round(ia_time)))
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
    
    def _load_models(self, path):
        graph = tf.Graph()
        with graph.as_default():
            session = tf.compat.v1.Session()
            with session.as_default():            
                model = load_model(path)
        return graph, session, model
    
    @staticmethod
    def _graph_timeline(log) -> None:
        time_series = log.copy()[['caseid', 'timestamp']]
        time_series['occ'] = 1
        time_series.set_index('timestamp', inplace=True)
        time_series.occ.rolling('3h').sum().plot(figsize=(30,10), linewidth=5, fontsize=10)
        plt.xlabel('Days', fontsize=20);
