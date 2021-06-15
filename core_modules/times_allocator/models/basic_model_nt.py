# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 10:15:12 2019

@author: Manuel Camargo
"""
# import keras.backend as K

import os
# import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
try:
    from support_modules.callbacks import time_callback as tc
except:
    from importlib import util
    spec = util.spec_from_file_location(
        'time_callback', 
        os.path.join(os.getcwd(), 'support_modules', 'callbacks', 'time_callback.py'))
    tc = util.module_from_spec(spec)
    spec.loader.exec_module(tc)


def _training_model(ac_weights, train_vec, valdn_vec, parms):
    """Example function with types documented in the docstring.
    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.
    Returns:
        bool: The return value. True for success, False otherwise.
    """

    print('Build model...')
# =============================================================================
#     Input layer
# =============================================================================
    ac_input = Input(shape=(train_vec['pref']['ac_index'].shape[1], ), 
                     name='ac_input')
    n_ac_input = Input(shape=(train_vec['pref']['n_ac_index'].shape[1], ), 
                     name='n_ac_input')
    features = Input(shape=(train_vec['pref']['features'].shape[1],
                           train_vec['pref']['features'].shape[2]), 
                     name='features')

# =============================================================================
#    Embedding layer for categorical attributes
# =============================================================================
    ac_embedding = Embedding(ac_weights.shape[0],
                             ac_weights.shape[1],
                             weights=[ac_weights],
                             input_length=train_vec['pref']['ac_index'].shape[1],
                             trainable=False, name='ac_embedding')(ac_input)
    n_ac_embedding = Embedding(ac_weights.shape[0],
                             ac_weights.shape[1],
                             weights=[ac_weights],
                             input_length=train_vec['pref']['n_ac_index'].shape[1],
                             trainable=False, name='n_ac_embedding')(n_ac_input)

# =============================================================================
#    Layer 1
# =============================================================================

    merged = Concatenate(name='concatenated', axis=2)(
        [ac_embedding, n_ac_embedding, features])

    l1_c1 = LSTM(parms['l_size'],
                 kernel_initializer='glorot_uniform',
                 return_sequences=True,
                 dropout=0.2,
                 implementation=parms['imp'])(merged)

# =============================================================================
#    Batch Normalization Layer
# =============================================================================
    batch1 = BatchNormalization()(l1_c1)

# =============================================================================
# The layer specialized in prediction
# =============================================================================
    l2_c1 = LSTM(parms['l_size'],
                activation=parms['lstm_act'],
                kernel_initializer='glorot_uniform',
                return_sequences=False,
                dropout=0.2,
                implementation=parms['imp'])(batch1)

# =============================================================================
# Output Layer
# =============================================================================
    times_output = Dense(train_vec['next'].shape[1],
                        activation=parms['dense_act'],
                        kernel_initializer='glorot_uniform',
                        name='time_output')(l2_c1)

    model = Model(inputs=[ac_input, n_ac_input, features],
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

    early_stopping = EarlyStopping(monitor='val_loss', patience=50)
    cb = tc.TimingCallback(parms['output'])
    # clean_models = cm.CleanSavedModelsCallback(parms['output'], 2)

    # Output file
    output_file_path = os.path.join(parms['output'], 
                                    parms['file'].split('.')[0] +
                                    '.h5')
    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
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

    batch_size = parms['batch_size']
    model.fit({'ac_input': train_vec['pref']['ac_index'],
               'n_ac_input': train_vec['pref']['n_ac_index'],
               'features': train_vec['pref']['features']},
              {'time_output': train_vec['next']},
              validation_data=(
                  {'ac_input': valdn_vec['pref']['ac_index'],
                   'n_ac_input': valdn_vec['pref']['n_ac_index'],
                   'features': valdn_vec['pref']['features']},
                {'time_output': valdn_vec['next']}),
              verbose=2,
              callbacks=[early_stopping, model_checkpoint, lr_reducer, cb],
              batch_size=batch_size,
              epochs=parms['epochs'])
    return model

