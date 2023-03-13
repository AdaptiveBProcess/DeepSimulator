# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:34:07 2021

@author: Manuel Camargo
"""
import os
import json
from datetime import datetime

import utils.support as sup

import tensorflow as tf

from tensorflow.keras.models import load_model

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.models import save_model


class TimesPredictorUpdater:
    """
    """

    def __init__(self, embbeding, modif_params, model_path):
        self.embbeding = embbeding
        self.modif_params = modif_params
        self.model_path = model_path

    def execute_pipeline(self):
        tf.compat.v1.reset_default_graph()
        # Update processing time predictive model
        self.update_model('_dpiapr', '_diapr',
                          ('lstm', 'batch_normalization', 'lstm_1'))
        # Update waiting time predictive model
        self.update_model('_dwiapr', '_diapr',
                          ('lstm_2', 'batch_normalization_1', 'lstm_3'))
        # Update metadata
        self.update_metadata()

    def update_metadata(self):
        # Read metadata data
        with open(self.model_path + '_diapr_meta.json') as file:
            model_metadata = json.load(file)
            model_metadata['ac_index'] = self.modif_params['ac_index']
            model_metadata['generated_at'] = (
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            model_metadata['roles_table'] += self.modif_params['m_tasks_assignment']
            sup.create_json(model_metadata, self.model_path + '_upd_diapr_meta.json')

    def update_model(self, model_ext, meta_ext, layer_names):
        # Read old predictive model
        pred_model_weights, model_parms = self.read_predictive_model(
            model_ext,
            meta_ext)
        # Create new predictive model
        new_pred_model = self.create_pred_model(model_parms, layer_names)
        # Re-load params
        for k, v in pred_model_weights.items():
            if v and k != 'ac_embedding':
                new_pred_model.get_layer(k).set_weights(v)
        # Save new model
        save_model(new_pred_model,
                   self.model_path + '_upd' + model_ext + '.h5',
                   save_format='h5')

    def read_predictive_model(self, model_ext, parms_exts):
        # Load old model
        model = load_model(self.model_path + model_ext + '.h5')
        model.summary()
        model_weights = {
            layer.name: layer.get_weights() for layer in model.layers}

        # Read metadata data
        with open(self.model_path + parms_exts + '_meta.json') as file:
            parameters = json.load(file)

        parameters = {
            **parameters,
            **{'ac_input_shape': model.get_layer('ac_input').input_shape[0],
               'features_shape': model.get_layer('features').input_shape[0],
               'ac_emb_inp_lenght': model.get_layer('ac_embedding').input_length,
               'time_output_lenght': model.get_layer('time_output').units,
               'imp': 1}}
        return model_weights, parameters

    def create_pred_model(self, parms, names):
        # Input layer
        ac_input = Input(shape=(parms['ac_input_shape'][1],), name='ac_input')
        features = Input(shape=(parms['features_shape'][1],
                                parms['features_shape'][2]),
                         name='features')

        # Embedding layer for categorical attributes
        ac_embedding = Embedding(self.embbeding.shape[0],
                                 self.embbeding.shape[1],
                                 weights=[self.embbeding],
                                 input_length=parms['ac_emb_inp_lenght'],
                                 trainable=False, name='ac_embedding')(ac_input)

        # Layer 1
        merged = Concatenate(name='concatenated', axis=2)([ac_embedding, features])

        l1_c1 = LSTM(parms['l_size'],
                     kernel_initializer='glorot_uniform',
                     return_sequences=True,
                     dropout=0.2,
                     implementation=parms['imp'],
                     name=names[0])(merged)

        # Batch Normalization Layer
        batch1 = BatchNormalization(name=names[1])(l1_c1)

        # The layer specialized in prediction
        l2_c1 = LSTM(parms['l_size'],
                     activation=parms['lstm_act'],
                     kernel_initializer='glorot_uniform',
                     return_sequences=False,
                     dropout=0.2,
                     implementation=parms['imp'],
                     name=names[2])(batch1)

        # Output Layer
        times_output = Dense(parms['time_output_lenght'],
                             activation=parms['dense_act'],
                             kernel_initializer='glorot_uniform',
                             name='time_output')(l2_c1)

        model = Model(inputs=[ac_input, features],
                      outputs=[times_output])

        if parms['optim'] == 'Nadam':
            opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        elif parms['optim'] == 'Adam':
            opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif parms['optim'] == 'SGD':
            opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
        elif parms['optim'] == 'Adagrad':
            opt = Adagrad(learning_rate=0.01)

        model.compile(loss={'time_output': 'mae'}, optimizer=opt)

        model.summary()
        return model
