# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:45:36 2020

@author: Manuel Camargo
"""
import os
import json
import copy
import shutil
import subprocess
import itertools
from operator import itemgetter
import pandas as pd
from xml.dom import minidom

from datetime import datetime

import utils.support as sup
from utils.support import safe_exec
from core_modules.sequences_generator import structure_optimizer as so

class SeqGenerator():
    """
    Main class of the Simulator
    """
    def __init__(self, parms, log_train):
        """constructor"""
        self.parms = parms
        self.log_train = log_train
        self.is_safe = True
        self.model_metadata = dict()
        self._execute_pipeline()

    def _execute_pipeline(self) -> None:
        model_path = os.path.join(
            self.parms['bpmn_models'],
            self.parms['file'].split('.')[0]+'.bpmn')
        if os.path.exists(model_path) and self.parms['update_gen']:
            print('Existe y actualizo!')
            self.is_safe = self._discover_model(True, is_safe=self.is_safe)
        elif not os.path.exists(model_path):
            print('No Existe!')
            self.is_safe = self._discover_model(False, is_safe=self.is_safe)
        self.model = model_path

    def generate(self, num_inst, start_time):
        self._modify_simulation_model(self.model, num_inst, start_time)
        self._generate_traces()

    @safe_exec
    def _discover_model(self, compare, **kwargs):
        structure_optimizer = so.StructureOptimizer(
            self.parms,
            copy.deepcopy(self.log_train))
        structure_optimizer.execute_trials()
        struc_model = structure_optimizer.best_output
        best_parms = structure_optimizer.best_parms
        best_similarity = structure_optimizer.best_similarity
        metadata_file = os.path.join(self.parms['bpmn_models'],
            self.parms['file'].split('.')[0]+'_meta.json')
        # compare with existing model
        save = True
        if compare:
            # Loading of parameters from existing model
            if os.path.exists(metadata_file):
                with open(metadata_file) as file:
                    data = json.load(file)
                    data = {k: v for k, v in data.items()}
                    print(data['similarity'])
                if data['similarity'] > best_similarity:
                    save = False
                    print('dont save')
        if save:
            # best structure mining parameters
            self.model_metadata['alg_manag'] = (
                self.parms['alg_manag'][best_parms['alg_manag']])
            self.model_metadata['gate_management'] = (
                self.parms['gate_management'][best_parms['gate_management']])
            if self.parms['mining_alg'] == 'sm1':
                self.model_metadata['epsilon'] = best_parms['epsilon']
                self.model_metadata['eta'] = best_parms['eta']
            elif self.parms['mining_alg'] == 'sm2':
                self.model_metadata['concurrency'] = best_parms['concurrency']
            self.model_metadata['similarity'] = best_similarity
            self.model_metadata['generated_at'] = (
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            # Copy best model to destination folder
            destination = os.path.join(self.parms['bpmn_models'],
                self.parms['file'].split('.')[0]+'.bpmn')
            source = os.path.join(struc_model,
                self.parms['file'].split('.')[0]+'.bpmn')
            shutil.copyfile(source, destination)
            # Save metadata
            sup.create_json(self.model_metadata, metadata_file)
        # clean output folder
        shutil.rmtree(structure_optimizer.temp_output)


    def _generate_traces(self):
        temp_path = self._temp_path_creation()
        # generate instances
        sim_log = self._execute_simulator(self.parms['bimp_path'],
                                          temp_path, self.model)
        # order by caseid and add position in the trace
        sim_log = pd.read_csv(sim_log)
        sim_log = self._sort_log(sim_log)
        sim_log.drop(columns=['start_timestamp', 'end_timestamp'], inplace=True)
        sim_log['caseid'] = sim_log['caseid'] + 1
        sim_log['caseid'] = sim_log['caseid'].astype('string')
        sim_log['caseid'] = 'Case'+sim_log['caseid']
        # save traces
        self.gen_seqs = sim_log
        # remove simulated log
        shutil.rmtree(temp_path)

    @staticmethod
    def _temp_path_creation() -> None:
        # Paths redefinition
        temp_path = os.path.join('output_files', sup.folder_id())
        # Output folder creation
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        return temp_path

    @staticmethod
    def _modify_simulation_model(model, num_inst, start_time):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        mydoc = minidom.parse(model)
        items = mydoc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(num_inst)
        items[0].attributes['startDateTime'].value = start_time
        # new_model_path = os.path.join(self.settings['output'],
        #                               os.path.split(model)[1])
        with open(model, 'wb') as f:
            f.write(mydoc.toxml().encode('utf-8'))
        f.close()
        # return new_model_path

    @staticmethod
    def _execute_simulator(bimp_path, temp_path, model):
        """Executes BIMP Simulations.
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        sim_log = os.path.join(temp_path, sup.file_id('SIM_'))
        args = ['java', '-jar', bimp_path, model, '-csv', sim_log]
        subprocess.run(args, check=True, stdout=subprocess.PIPE)
        return sim_log

    @staticmethod
    def _sort_log(log):
        log = sorted(log.to_dict('records'), key=lambda x: x['caseid'])
        for key, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('start_timestamp'))
            length = len(events)
            for i in range(0, len(events)):
                events[i]['pos_trace'] = i + 1
                events[i]['trace_len'] = length
        log = pd.DataFrame.from_dict(log)
        log.sort_values(by='start_timestamp', ascending=True, inplace=True)
        return log
