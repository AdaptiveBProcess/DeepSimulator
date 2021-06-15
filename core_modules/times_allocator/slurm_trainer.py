# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 12:01:20 2021

@author: Manuel Camargo
"""
import os
import sys
import csv
import json
import getopt

import numpy as np
import pandas as pd

import utils.support as sup
import tensorflow as tf

import samples_creator as sc
from models import basic_model as bsc
from models import dual_model as dual
from models import basic_model_nt as innt

class SlurmWorker():
    """
    Hyperparameter-optimizer class
    """
    def __init__(self, argv):
        """constructor"""
        self.parms = dict()
        try:
            opts, _ = getopt.getopt( argv, "h:p:f:r:", 
                                    ['parms_file=', 
                                     'output_folder=', 
                                     'res_files='])
            for opt, arg in opts:
                key = self.catch_parameter(opt)
                self.parms[key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
        # Output Files
        self.temp_output = os.path.join(os.getcwd(),
                                        self.parms['output_folder'])
        self.res_files = os.path.join(os.getcwd(),self.parms['res_files'])
        self.load_parameters()
        column_names = {'Case ID': 'caseid',
                        'Activity': 'task',
                        'lifecycle:transition': 'event_type',
                        'Resource': 'user'}
        self.parms['one_timestamp'] = False  # Only one timestamp in the log
        self.parms['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': self.parms['one_timestamp'],
            'filter_d_attrib': False}
        print(self.parms)
        self.log_train = pd.read_csv(
            os.path.join(self.temp_output, 'opt_parms', 'log_train.csv'))
        self.log_valdn = pd.read_csv(
            os.path.join(self.temp_output, 'opt_parms', 'log_valdn.csv'))
        self.read_embeddings(self.parms)
        loss = self.exec_pipeline()
        self._define_response(self.parms, loss)
        print('COMPLETED')


    @staticmethod
    def catch_parameter(opt):
        """Change the captured parameters names"""
        switch = {'-h': 'help', '-p': 'parms_file', 
                  '-f': 'output_folder', '-r': 'res_files'}
        return switch[opt]

    def load_parameters(self):
        # Loading of parameters from training
        path = os.path.join(self.temp_output, 
                            'opt_parms',
                            self.parms['parms_file'])
        with open(path) as file:
            data = json.load(file)
            parms = {k: v for k, v in data.items()}
            self.parms = {**self.parms, **parms}
            self.ac_index = {k: int(v) for k, v in data['ac_index'].items()}
            file.close()
        self.index_ac = {v: k for k, v in self.ac_index.items()}
    
    def read_embeddings(self, params):
        # Load embedded matrix
        ac_emb_name = 'ac_' + params['file'].split('.')[0]+'.emb'
        if os.path.exists(os.path.join(os.getcwd(),
                                       'input_files',
                                       'embedded_matix',
                                        ac_emb_name)):
            self.ac_weights = self.load_embedded(self.index_ac, ac_emb_name)

    def exec_pipeline(self):
        print(self.parms)
        # Path redefinition
        self.parms = self._temp_path_redef(self.parms)
        # Vectorize input
        vectorizer = sc.SequencesCreator(self.parms['read_options']['one_timestamp'], 
                                         self.ac_index)
        train_vec = vectorizer.vectorize(self.parms['model_type'],
                                          self.log_train,
                                          self.parms)
        valdn_vec = vectorizer.vectorize(self.parms['model_type'], 
                                          self.log_valdn, 
                                          self.parms)
        # Train
        trainer = self._get_trainer(self.parms['model_type'])
        tf.compat.v1.reset_default_graph()
        model = trainer(self.ac_weights,
                        train_vec,
                        valdn_vec,
                        self.parms)
        # evaluation
        acc = self.evaluate_model(
            self.parms['model_type'], model, valdn_vec)
            
        rsp = self._define_response(self.parms, acc['loss'])
        print("-- End of trial --")
        return rsp

    def evaluate_model(self, model_type, model, valdn_vec):
        if model_type in ['inter', 'basic']:
            return model.evaluate(
                x={'ac_input': valdn_vec['pref']['ac_index'],
                   'features': valdn_vec['pref']['features']},
                y={'time_output': valdn_vec['next']['expected']},
                return_dict=True)
        elif model_type == 'inter_nt':
            return model.evaluate(
                x={'ac_input': valdn_vec['pref']['ac_index'],
                   'n_ac_input': valdn_vec['pref']['n_ac_index'],
                   'features': valdn_vec['pref']['features']},
                y={'time_output': valdn_vec['next']},
                return_dict=True)
        elif model_type == 'dual_inter':
            # with model['proc_model']['graph'].as_default():
            acc_proc = model['proc_model']['model'].evaluate(
                x={'ac_input': valdn_vec['proc_model']['pref']['ac_index'],
                   'features': valdn_vec['proc_model']['pref']['features']},
                y={'time_output': valdn_vec['proc_model']['next']},
                return_dict=True)
            # with model['wait_model']['graph'].as_default():
            acc_wait = model['wait_model']['model'].evaluate(
                x={'ac_input': valdn_vec['waiting_model']['pref']['ac_index'],
                   'features': valdn_vec['waiting_model']['pref']['features']},
                y={'time_output': valdn_vec['waiting_model']['next']},
                return_dict=True)
            return {'loss': (0.5*acc_proc['loss']) + (0.5*acc_wait['loss'])}
        else:
            raise ValueError('Unexistent model')


    def _temp_path_redef(self, settings, **kwargs) -> dict:
        # Paths redefinition
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())
        # Output folder creation
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
        return settings

    def _define_response(self, parms, loss, **kwargs) -> None:
        measurements = list()
        measurements.append({'loss': loss,
                             'sim_metric': 'val_loss',
                             'n_size': parms['n_size'],
                             'l_size': parms['l_size'],
                             'lstm_act': parms['lstm_act'],
                             'dense_act': parms['dense_act'],
                             'optim': parms['optim'],
                             'output': parms['output'][len(os.getcwd())+1:]})
        if os.path.getsize(self.res_files) > 0:
            sup.create_csv_file(measurements, self.res_files, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.res_files)

    def _get_trainer(self, model_type):
        if model_type in ['basic', 'inter']:
            return bsc._training_model
        elif model_type == 'inter_nt':
            return innt._training_model
        elif model_type == 'dual_inter':
            return dual._training_model
        else:
            raise ValueError(model_type)


    @staticmethod
    def load_embedded(index, filename):
        """Loading of the embedded matrices.
        parms:
            index (dict): index of activities or roles.
            filename (str): filename of the matrix file.
        Returns:
            numpy array: array of weights.
        """
        weights = list()
        input_folder = os.path.join(os.getcwd(), 'input_files', 'embedded_matix')
        with open(os.path.join(input_folder, filename), 'r') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in filereader:
                cat_ix = int(row[0])
                if index[cat_ix] == row[1].strip():
                    weights.append([float(x) for x in row[2:]])
            csvfile.close()
        return np.array(weights)


if __name__ == "__main__":
    worker = SlurmWorker(sys.argv[1:])
