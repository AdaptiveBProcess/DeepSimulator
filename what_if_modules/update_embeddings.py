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

class ModelUpdater():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, parms):
        self.parms = parms


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
    # model, ac_weights, org_parms = read_original_model(parms, parms['gl']['file'])
    
    # Load expected data
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

