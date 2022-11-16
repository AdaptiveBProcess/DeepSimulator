# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:05:36 2021

@author: Manuel Camargo
"""
import os
import copy
import random
import itertools
import traceback

import pandas as pd
import utils.support as sup
import utils.slurm_multiprocess as slmp


class ModelHPCOptimizer():
    """
    Hyperparameter-optimizer class
    """
       
    def __init__(self, parms, log_train, log_valdn, ac_index, ac_weights):
        """constructor"""
        self.space = self.define_search_space(parms)
        """self.log_train = copy.deepcopy(log_train)
        self.log_valdn = copy.deepcopy(log_valdn)
        self.ac_index = ac_index
        self.ac_weights = ac_weights
        # Load settings
        self.parms = parms
        self.temp_output = parms['output']
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
            os.makedirs(os.path.join(self.temp_output, 'opt_parms'))
        self.file_name = os.path.join(self.temp_output, sup.file_id(prefix='OP_'))
        # Results file
        if not os.path.exists(self.file_name):
            open(self.file_name, 'w').close()
        """
        self.conn = {'partition': 'main',
                    'mem': str(32000),
                    'cpus': str(10),
                    'env': 'deep_sim3',
                    'script': os.path.join('core_modules', 
                                           'times_allocator',
                                           'slurm_trainer.py')}
        self.slurm_workers = 50
        self.best_output = None
        self.best_parms = dict()
        self.best_loss = 1
        
    """
    @staticmethod
    def define_search_space(parms):
        space = list()
        listOLists = [parms['lstm_act'], 
                      parms['dense_act'], 
                      parms['n_size'],
                      parms['l_size'], 
                      parms['optim']]
        # selection method definition
        preconfigs = list()
        for lists in itertools.product(*listOLists):
            preconfigs.append(dict(lstm_act=lists[0],
                                   dense_act=lists[1],
                                   n_size=lists[2],
                                   l_size=lists[3],
                                   optim=lists[4]))
        def_parms = {
            'imp': parms['imp'], 'file': parms['file'],
            'batch_size': parms['batch_size'], 'epochs': parms['epochs']}
        for config in random.sample(preconfigs, parms['max_eval']):
            space.append({**config, **def_parms})
        return space
    def export_params(self):
        configs_files = list()
        for config in self.space:
            config['ac_index'] = self.ac_index
            config['model_type'] = self.parms['model_type']
            config['all_r_pool'] = self.parms['all_r_pool']
            conf_file = sup.file_id(prefix='CNF_', extension='.json')
            sup.create_json(
                config, os.path.join(self.temp_output, 'opt_parms', conf_file))
            configs_files.append(conf_file)
        self.log_train.to_csv(
            os.path.join(self.temp_output, 'opt_parms', 'log_train.csv'),
            index=False, encoding='utf-8')
        self.log_valdn.to_csv(
            os.path.join(self.temp_output, 'opt_parms', 'log_valdn.csv'),
            index=False, encoding='utf-8')
        return configs_files
    """

    def execute_trials(self):
        #configs_files = self.export_params()
        """args = [{'p': config, 
                 'f': self.temp_output,
                 'r': self.file_name} for config in configs_files]
        """
        mprocessor = slmp.HPC_Multiprocess(self.conn,
                                            args,
                                            self.temp_output,
                                            None,
                                            self.slurm_workers,
                                            timeout=5)
        mprocessor.parallelize()
        try:
            results = (pd.read_csv(self.file_name)
                       .sort_values('loss', ascending=True))
            result = results.head(1).iloc[0]
            self.best_output = result.output
            self.best_loss = result.loss
            self.best_parms = results.head(1).to_dict('records')[0]
        except Exception as e:
            print(e)
            traceback.print_exc()
            pass

