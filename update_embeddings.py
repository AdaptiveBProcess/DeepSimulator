# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:02:19 2021

@author: Manuel Camargo
"""
import os
import sys
import getopt
import pandas as pd
import numpy as np
import json
import math
import random
from datetime import datetime

import utils.support as sup

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization, Dot, Reshape
from tensorflow.keras.optimizers import Nadam, Adam, SGD, Adagrad
from tensorflow.keras.models import save_model

# =============================================================================
# Embedding methods
# =============================================================================

def read_embedded(filename):
    """Loading of the embedded matrices.
    parms:
        index (dict): index of activities or roles.
        filename (str): filename of the matrix file.
    Returns:
        numpy array: array of weights.
    """
    weights = pd.read_csv(filename, header=None)
    weights[1] = weights.apply(lambda x: x[1].strip(), axis=1)
    weights = weights.drop(columns=[0, 1])
    return weights.to_numpy()

def reformat_matrix(index, weigths):
    """Reformating of the embedded matrix for exporting.
    Args:
        index: index of activities or users.
        weigths: matrix of calculated coordinates.
    Returns:
        matrix with indexes.
    """
    matrix = list()
    for i, _ in enumerate(index):
        data = [i, index[i]]
        data.extend(weigths[i])
        matrix.append(data)
    return matrix

def create_embedding_model(num_categories, num_users, embedding_size):
    """Model to embed activities and users using the functional API"""

    # Both inputs are 1-dimensional
    activity = Input(name='activity', shape=[1])
    user = Input(name='user', shape=[1])

    # Embedding the activity (shape will be (None, 1, embedding_size))
    activity_embedding = Embedding(name='activity_embedding',
                                   input_dim=num_categories,
                                   output_dim=embedding_size)(activity)

    # Embedding the user (shape will be (None, 1, embedding_size))
    user_embedding = Embedding(name='user_embedding',
                               input_dim=num_users,
                               output_dim=embedding_size)(user)

    # Merge the layers with a dot product
    # along the second axis (shape will be (None, 1, 1))
    merged = Dot(name='dot_product',
                 normalize=True, axes=2)([activity_embedding, user_embedding])

    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape=[1])(merged)

    # Loss function is mean squared error
    model = Model(inputs=[activity, user], outputs=merged)
    model.compile(optimizer='Adam', loss='mse')

    return model

def read_original_model(parms, event_log):
    # Load old model
    model_path = os.path.join(parms['gl']['embedded_path'],
                        event_log.split('.')[0]+'_emb.h5')
    model = load_model(model_path)
    model.summary()
    # Load embeddings    
    emb_path = os.path.join(parms['gl']['embedded_path'],
                         'ac_'+event_log.split('.')[0]+'.emb')
    ac_weights = read_embedded(emb_path)
    # Read data
    data_path = os.path.join(parms['gl']['times_gen_path'],
                         event_log.split('.')[0]+'_diapr_meta.json')
    with open(data_path) as file:
        parameters = json.load(file)
    return model, ac_weights, parameters

def re_train_embedded(modif_params, model, dim_number):
    """Carry out the training of the embeddings"""
    # Iterate through each book
    vec, cl = vectorize_input(modif_params, negative_ratio=2)
    
    # Output file
    output_file_path = os.path.join(modif_params['embedded_path'],
                                    modif_params['file'].split('.')[0]+
                                    '_upd_emb.h5')
    # Saving
    model_checkpoint = ModelCheckpoint(output_file_path,
                                        monitor='val_loss',
                                        verbose=0,
                                        save_best_only=True,
                                        save_weights_only=False,
                                        mode='auto')
    # Train
    model.fit(x=vec, y=cl,
              validation_split=0.2,
              callbacks=[model_checkpoint],
              epochs=100,
              verbose=2)

    # Extract embeddings
    return model.get_layer('activity_embedding').get_weights()[0]


def vectorize_input(modif_params, negative_ratio=1.0):
    """Generate batches of samples for training"""
    pairs = list()
    roles = (pd.DataFrame.from_dict(modif_params['roles'], orient='index')
             .unstack()
             .reset_index()
             .drop(columns='level_0')
             .rename(columns={'level_1': 'role', 0: 'user'}))
    roles = roles[~roles.user.isna()]
    # Task assignments
    m_tasks_assignment = pd.DataFrame(modif_params['m_tasks_assignment'])
    roles = roles.merge(m_tasks_assignment, on='role', how='left')
    pairs = roles[['task','user']][~roles.task.isna()].to_records(index=False)
    pairs = [
        (modif_params['ac_index'][x[0]], modif_params['usr_index'][x[1]]) for x in pairs]
        
    n_positive = math.ceil(modif_params['len_log'] * 0.15)
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    pairs_set = set(pairs)
    activities = modif_params['m_tasks']
    users = list(modif_params['usr_index'].keys())
    # This creates a generator
    # randomly choose positive examples
    idx = 0
    for idx in range(n_positive):
        activity, user = random.sample(pairs, 1)[0]
        batch[idx, :] = (activity, user, 1)
    # Increment idx by 1
    idx += 1

    # Add negative examples until reach batch size
    while idx < batch_size:
        # random selection
        random_ac = modif_params['ac_index'][random.sample(activities, 1)[0]]
        random_rl = random.randrange(len(users)-1)

        # Check to make sure this is not a positive example
        if (random_ac, random_rl) not in pairs_set:

            # Add to batch and increment index,  0 due classification task
            batch[idx, :] = (random_ac, random_rl, 0)
            idx += 1
    # Make sure to shuffle order
    np.random.shuffle(batch)
    return {'activity': batch[:, 0], 'user': batch[:, 1]}, batch[:, 2]

# =============================================================================
# Predictive models
# =============================================================================
def read_predictive_model(parms, model_ext, parms_exts):
    # Load old model
    model_path = os.path.join(parms['gl']['times_gen_path'],
                              parms['gl']['file'].split('.')[0]+model_ext+'.h5')
    model = load_model(model_path)
    model.summary()
    model_weights = {layer.name: layer.get_weights() for layer in model.layers}
    
    # Read metadata data
    data_path = os.path.join(
        parms['gl']['times_gen_path'],
        parms['gl']['file'].split('.')[0]+parms_exts+'_meta.json')
    
    with open(data_path) as file:
        parameters = json.load(file)

    parameters = {
        **parameters,
        **{'ac_input_shape': model.get_layer('ac_input').input_shape[0],
           'features_shape': model.get_layer('features').input_shape[0],
           'ac_emb_inp_lenght': model.get_layer('ac_embedding').input_length,
           'time_output_lenght': model.get_layer('time_output').units,
           'imp': 1}}
    return model_weights, parameters


def create_pred_model(ac_weights, parms, names):
    # Input layer
    ac_input = Input(shape=(parms['ac_input_shape'][1], ), name='ac_input')
    features = Input(shape=(parms['features_shape'][1], 
                            parms['features_shape'][2]),
                     name='features')

    # Embedding layer for categorical attributes
    ac_embedding = Embedding(ac_weights.shape[0],
                             ac_weights.shape[1],
                             weights=[ac_weights],
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


# =============================================================================
# General methods
# =============================================================================
def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


def main(argv):
    """Main aplication method"""
    parms = dict()
    parms = define_general_parms(parms)
    # parms setting manual fixed or catched by console
    if not argv:
        # Event-log parms
        parms['gl']['file'] = 'cvs_1000_modified.xes'
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt( argv, "h:f:",['file='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parms['gl'][key] = None
                else:
                    parms['gl'][key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    # Modification parameters
    modif_params = dict()
    # Load old model
    model, ac_weights, org_parms = read_original_model(parms, parms['gl']['file'])
    
    # Load expected data
    data_path = os.path.join(parms['gl']['times_gen_path'],
                         'cvs_1000_complete.xes'.split('.')[0]+'_diapr_meta.json')
    with open(data_path) as file:
        exp_parms = json.load(file)
    # ac and roles missings comparison
    m_tasks = list(set(exp_parms['ac_index'].keys()) - set(org_parms['ac_index'].keys()))
    m_tasks_assignment = list(filter(lambda x: x['task'] in m_tasks, exp_parms['roles_table']))
    # Save params
    modif_params['ac_index'] = org_parms['ac_index']
    modif_params['usr_index'] = org_parms['usr_index']
    modif_params['m_tasks'] = m_tasks
    modif_params['m_tasks_assignment'] = m_tasks_assignment
    modif_params['roles'] = exp_parms['roles']
    modif_params['len_log'] = 1000
    modif_params['embedded_path'] = parms['gl']['embedded_path']
    modif_params['file'] = parms['gl']['file']
    # extend indexes
    for task in m_tasks: 
        modif_params['ac_index'][task] = len(modif_params['ac_index'])
    # extend embedding
    new_ac_weights = np.append(ac_weights, np.random.rand(len(m_tasks), ac_weights.shape[1]), axis=0)
    
    # Create new model
    output_shape = model.get_layer('activity_embedding').output_shape
    users_input_dim = model.get_layer('user_embedding').input_dim
    num_emb_dimmensions = output_shape[2]
    new_model = create_embedding_model(new_ac_weights.shape[0], 
                                       users_input_dim, 
                                       num_emb_dimmensions)
    # Save weights of old model
    temp_weights = {layer.name: layer.get_weights() for layer in model.layers}
    # load values in new model
    for k, v in temp_weights.items():
        if k == 'activity_embedding':
            new_model.get_layer(k).set_weights([new_ac_weights])
        elif v:
            new_model.get_layer(k).set_weights(v)
    
    # create new examples
    ac_weights = re_train_embedded(modif_params, new_model, num_emb_dimmensions)
    
    matrix = reformat_matrix({v: k for k, v in org_parms['ac_index'].items()},
                             ac_weights)
    sup.create_file_from_list(
        matrix,
        os.path.join(parms['gl']['embedded_path'],
                     'ac_' + parms['gl']['file'].split('.')[0]+'_upd.emb'))
    ###################################
    # Read old predictive model
    pred_model_weights, model_parms = read_predictive_model(parms, '_dpiapr', '_diapr')
    # Create new predictive model
    new_pred_model = create_pred_model(ac_weights, model_parms, 
                                       ('lstm', 'batch_normalization','lstm_1'))
    # Re-load params
    for k, v in pred_model_weights.items():
        if v and k != 'ac_embedding':
            new_pred_model.get_layer(k).set_weights(v)
    save_model(new_pred_model, 
               os.path.join(parms['gl']['times_gen_path'],
                            parms['gl']['file'].split('.')[0]+'_upd_dpiapr.h5'),
               save_format='h5')
               
    #------------
    # Read old predictive model
    pred_model_weights, model_parms = read_predictive_model(parms, '_dwiapr', '_diapr')
    # Create new predictive model
    new_pred_model = create_pred_model(ac_weights, model_parms, 
                                       ('lstm_2', 'batch_normalization_1', 'lstm_3'))
    # Re-load params
    for k, v in pred_model_weights.items():
        if v and k != 'ac_embedding':
            new_pred_model.get_layer(k).set_weights(v)
    save_model(new_pred_model, 
               os.path.join(parms['gl']['times_gen_path'],
                            parms['gl']['file'].split('.')[0]+'_upd_dwiapr.h5'),
               save_format='h5')
    ###################################
    # Read metadata data
    data_path = os.path.join(
        parms['gl']['times_gen_path'],
        parms['gl']['file'].split('.')[0]+'_diapr_meta.json')
    with open(data_path) as file:
        model_metadata = json.load(file)
        model_metadata['ac_index'] = modif_params['ac_index']
        model_metadata['generated_at'] = (
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        model_metadata['roles_table'] += modif_params['m_tasks_assignment']
        
        data_path = os.path.join(
            parms['gl']['times_gen_path'],
            parms['gl']['file'].split('.')[0]+'_upd_diapr_meta.json')
        sup.create_json(model_metadata, data_path)
    
def define_general_parms(parms):
    """ Sets the app general parms"""
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    parms['gl'] = dict()
    # Event-log reading options
    parms['gl']['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': False,
            'filter_d_attrib': True}
    # Folders structure
    parms['gl']['event_logs_path'] = os.path.join('input_files', 
                                                  'event_logs')
    parms['gl']['bpmn_models'] = os.path.join('input_files', 
                                              'bpmn_models')
    parms['gl']['embedded_path'] = os.path.join('input_files', 
                                                'embedded_matix')
    parms['gl']['times_gen_path'] = os.path.join('input_files', 
                                                 'times_gen_models')
    return parms


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])

