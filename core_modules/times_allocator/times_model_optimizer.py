# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:48:57 2020

@author: Manuel Camargo
"""
import os
import copy
import traceback
import pandas as pd
from hyperopt import tpe
from hyperopt import Trials, hp, fmin, STATUS_OK, STATUS_FAIL

import utils.support as sup

import tensorflow as tf
from core_modules.times_allocator import samples_creator as sc
from core_modules.times_allocator.models import basic_model as bsc
from core_modules.times_allocator.models import dual_model as dual
from core_modules.times_allocator.models import basic_model_nt as innt



class TimesModelOptimizer():
    """
    Hyperparameter-optimizer class
    """
    class Decorators(object):

        @classmethod
        def safe_exec(cls, method):
            """
            Decorator to safe execute methods and return the state
            ----------
            method : Any method.
            Returns
            -------
            dict : execution status
            """
            def safety_check(*args, **kw):
                status = kw.get('status', method.__name__.upper())
                response = {'values': [], 'status': status}
                if status == STATUS_OK:
                    try:
                        response['values'] = method(*args)
                    except Exception as e:
                        print(e)
                        traceback.print_exc()
                        response['status'] = STATUS_FAIL
                return response
            return safety_check
        
    def __init__(self, parms, log_train, log_valdn, ac_index, ac_weights):
        """constructor"""
        self.space = self.define_search_space(parms)
        self.log_train = copy.deepcopy(log_train)
        self.log_valdn = copy.deepcopy(log_valdn)
        self.ac_index = ac_index
        self.ac_weights = ac_weights
        # Load settings
        self.parms = parms
        self.temp_output = parms['output']
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
        self.file_name = os.path.join(self.temp_output, sup.file_id(prefix='OP_'))
        # Results file
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()
        # Trials object to track progress
        self.bayes_trials = Trials()
        self.best_output = None
        self.best_parms = dict()
        self.best_loss = 1
        
    @staticmethod
    def define_search_space(parms):
        space = {'n_size': hp.choice('n_size', parms['n_size']),
                 'l_size': hp.choice('l_size', parms['l_size']),
                 'lstm_act': hp.choice('lstm_act', parms['lstm_act']),
                 'dense_act': hp.choice('dense_act', parms['dense_act']),
                 'optim': hp.choice('optim', parms['optim']),
                 'imp': parms['imp'], 'file': parms['file'],
                 'batch_size': parms['batch_size'], 'epochs': parms['epochs']}
        return space

    def execute_trials(self):

        def exec_pipeline(trial_stg):
            print(trial_stg)
            trial_stg['all_r_pool'] = self.parms['all_r_pool']
            status = STATUS_OK
            # Path redefinition
            rsp = self._temp_path_redef(trial_stg, status=status)
            status = rsp['status']
            trial_stg = rsp['values'] if status == STATUS_OK else trial_stg
            # Vectorize input
            vectorizer = sc.SequencesCreator(
                self.parms['read_options']['one_timestamp'], self.ac_index)
            train_vec = vectorizer.vectorize(
                self.parms['model_type'], self.log_train, trial_stg)
            valdn_vec = vectorizer.vectorize(
                self.parms['model_type'], self.log_valdn, trial_stg)
            # Train
            trainer = self._get_trainer(self.parms['model_type'])
            tf.compat.v1.reset_default_graph()
            model = trainer(self.ac_weights, train_vec, valdn_vec, trial_stg)
            # evaluation
            acc = self.evaluate_model(
                self.parms['model_type'], model, valdn_vec)
            print(acc)
            rsp = self._define_response(trial_stg, status, acc['loss'])
            print("-- End of trial --")
            return rsp

        # Optimize
        best = fmin(fn=exec_pipeline,
                    space=self.space,
                    algo=tpe.suggest,
                    max_evals=self.parms['max_eval'],
                    trials=self.bayes_trials,
                    show_progressbar=False)
        # Save results
        try:
            results = (pd.DataFrame(self.bayes_trials.results)
                       .sort_values('loss', ascending=True))
            result = results[results.status == 'ok'].head(1).iloc[0]
            self.best_output = result.output
            self.best_loss = result.loss
            self.best_parms = {k: self.parms[k][v] for k, v in best.items()}
        except Exception as e:
            print(e)
            pass

    def _get_trainer(self, model_type):
        if model_type in ['basic', 'inter']:
            return bsc._training_model
        elif model_type == 'inter_nt':
            return innt._training_model
        elif model_type == 'dual_inter':
            return dual._training_model
        else:
            raise ValueError(model_type)

    @Decorators.safe_exec
    def _temp_path_redef(self, settings, **kwargs) -> dict:
        # Paths redefinition
        settings['output'] = os.path.join(self.temp_output, sup.folder_id())
        # Output folder creation
        if not os.path.exists(settings['output']):
            os.makedirs(settings['output'])
        return settings

    def _define_response(self, parms, status, loss, **kwargs) -> None:
        print(loss)
        response = dict()
        measurements = list()
        data = {'n_size': parms['n_size'],
                'l_size': parms['l_size'],
                'lstm_act': parms['lstm_act'],
                'dense_act': parms['dense_act'],
                'optim': parms['optim']}
        response['output'] = parms['output']
        if status == STATUS_OK:
            response['loss'] = loss
            response['status'] = status if loss > 0 else STATUS_FAIL
            measurements.append({**{'loss': loss,
                                    'sim_metric': 'mae', 
                                    'status': response['status']},
                                 **data})
        else:
            response['status'] = status
            measurements.append({**{'loss': 1,
                                    'sim_metric': 'mae',
                                    'status': response['status']},
                                 **data})
        if os.path.getsize(self.file_name) > 0:
            sup.create_csv_file(measurements, self.file_name, mode='a')
        else:
            sup.create_csv_file_header(measurements, self.file_name)
        return response
    
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
    
