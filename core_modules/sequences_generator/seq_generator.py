# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:45:36 2020

@author: Manuel Camargo
"""
import copy
import itertools
import json
import os
import shutil
import subprocess
from abc import ABCMeta, abstractmethod
from datetime import datetime
from operator import itemgetter
from pathlib import Path
from xml.dom import minidom

import pandas as pd
import utils.support as sup
from utils.support import safe_exec

from core_modules.sequences_generator import structure_optimizer as so
from support_modules.common import FileExtensions as Fe
from support_modules.common import LogAttributes as La
from support_modules.common import SequencesGenerativeMethods as SqM


class SeqGeneratorFabric:

    @classmethod
    def get_generator(cls, method):
        if method == SqM.PROCESS_MODEL:
            return StochasticProcessModelGenerator
        elif method == SqM.TEST:
            return OriginalSequencesGenerator
        else:
            raise ValueError('Nonexistent sequences generator')


class SeqGenerator(metaclass=ABCMeta):
    """
    Generator base class
    """

    def __init__(self, parameters, log_train):
        """constructor"""
        self.parameters = parameters
        self.log_train = log_train
        self.is_safe = True
        self.model_metadata = dict()
        self.gen_seqs = None

    @abstractmethod
    def generate(self, num_inst, start_time):
        pass

    @abstractmethod
    def clean_time_stamps(self):
        pass

    @staticmethod
    def sort_log(log):
        log = sorted(log.to_dict('records'), key=lambda x: x[La.CASE_ID])
        for key, group in itertools.groupby(log, key=lambda x: x[La.CASE_ID]):
            events = list(group)
            events = sorted(events, key=itemgetter(La.START_TIME))
            length = len(events)
            for i in range(0, len(events)):
                events[i]['pos_trace'] = i + 1
                events[i]['trace_len'] = length
        log = pd.DataFrame.from_records(log)
        log.sort_values(by=La.START_TIME, ascending=True, inplace=True)
        return log


class StochasticProcessModelGenerator(SeqGenerator):

    def generate(self, log, start_time):
        num_inst = len(log.caseid.unique())
        # verify if model exists
        self._verify_model()
        # update model parameters
        self._modify_simulation_model(self.model, num_inst, start_time)
        temp_path = self._temp_path_creation()
        # generate instances
        sim_log = self._execute_simulator(self.parameters['bimp_path'], temp_path, self.model)
        # order by Case ID and add position in the trace
        sim_log = self._rename_sim_log(sim_log)
        # save traces
        self.gen_seqs = sim_log
        # remove simulated log
        shutil.rmtree(temp_path)

    def _rename_sim_log(self, sim_log):
        sim_log = pd.read_csv(sim_log)
        sim_log = self.sort_log(sim_log)
        sim_log[La.CASE_ID] = sim_log[La.CASE_ID] + 1
        sim_log[La.CASE_ID] = sim_log[La.CASE_ID].astype('string')
        sim_log[La.CASE_ID] = f"Case{sim_log[La.CASE_ID]}"
        return sim_log

    def clean_time_stamps(self):
        self.gen_seqs.drop(columns=[La.START_TIME, La.END_TIME], inplace=True)

    def _verify_model(self) -> None:
        model_path = os.path.join(self.parameters['bpmn_models'], self.parameters['file'].split('.')[0] + Fe.BPMN)
        if os.path.exists(model_path) and self.parameters['update_gen']:
            self.is_safe = self._discover_model(True, is_safe=self.is_safe)
        elif not os.path.exists(model_path):
            self.is_safe = self._discover_model(False, is_safe=self.is_safe)
        self.model = model_path

    @safe_exec
    def _discover_model(self, compare, **_kwargs):
        structure_optimizer = so.StructureOptimizer(self.parameters, copy.deepcopy(self.log_train))
        structure_optimizer.execute_trials()
        struc_model = structure_optimizer.best_output
        best_parameters = structure_optimizer.best_parms
        best_similarity = structure_optimizer.best_similarity
        metadata_file = os.path.join(self.parameters['bpmn_models'],
                                     f"{self.parameters['file'].split('.')[0]}_meta{Fe.JSON}")
        # compare with existing model
        save = True
        if compare:
            # Loading of parameters from existing model
            save = self._loading_parameters_from_existing_model(best_similarity, metadata_file, save)
        if save:
            # best structure mining parameters
            self._extract_model_metadata(best_parameters, best_similarity)
            # Copy best model to destination folder
            self._copy_best_model(struc_model)
            # Save metadata
            sup.create_json(self.model_metadata, metadata_file)
        # clean output folder
        shutil.rmtree(structure_optimizer.temp_output)

    def _copy_best_model(self, struc_model):
        file_name = f"{self.parameters['file'].split('.')[0]}{Fe.BPMN}"
        destination = os.path.join(self.parameters['bpmn_models'], file_name)
        source = os.path.join(struc_model, file_name)
        shutil.copyfile(source, destination)

    def _extract_model_metadata(self, best_parameters, best_similarity):
        self.model_metadata['alg_manag'] = (self.parameters['alg_manag'][best_parameters['alg_manag']])
        self.model_metadata['gate_management'] = (self.parameters['gate_management'][best_parameters['gate_management']])
        if self.parameters['mining_alg'] == 'sm1':
            self.model_metadata['epsilon'] = best_parameters['epsilon']
            self.model_metadata['eta'] = best_parameters['eta']
        elif self.parameters['mining_alg'] == 'sm2':
            self.model_metadata['concurrency'] = best_parameters['concurrency']
        self.model_metadata['similarity'] = best_similarity
        self.model_metadata['generated_at'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    @staticmethod
    def _loading_parameters_from_existing_model(best_similarity, metadata_file, save):
        if os.path.exists(metadata_file):
            with open(metadata_file) as file:
                data = json.load(file)
                data = {k: v for k, v in data.items()}
                print(data['similarity'])
            if data['similarity'] > best_similarity:
                save = False
                print('dont save')
        return save

    @staticmethod
    def _temp_path_creation() -> Path:
        # Paths redefinition
        temp_path = os.path.join('output_files', sup.folder_id())
        # Output folder creation
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        return Path(temp_path)

    @staticmethod
    def _modify_simulation_model(model, num_inst, start_time):
        """Modifies the number of instances of the BIMP simulation model
        to be equal to the number of instances in the testing log"""
        my_doc = minidom.parse(model)
        items = my_doc.getElementsByTagName('qbp:processSimulationInfo')
        items[0].attributes['processInstances'].value = str(num_inst)
        items[0].attributes['startDateTime'].value = start_time
        with open(model, 'wb') as f:
            f.write(my_doc.toxml().encode('utf-8'))
        f.close()

    @staticmethod
    def _execute_simulator(bimp_path, temp_path, model):
        sim_log = os.path.join(temp_path, sup.file_id('SIM_'))
        args = ['java', '-jar', bimp_path, model, '-csv', sim_log]
        subprocess.run(args, check=True, stdout=subprocess.PIPE)
        return sim_log


class OriginalSequencesGenerator(SeqGenerator):

    def generate(self, log, start_time):
        sequences = log.copy(deep=True)
        sequences = sequences[[La.CASE_ID, La.ACTIVITY, La.RESOURCE, La.START_TIME]]
        replacements = {case_name: f'Case{idx + 1}' for idx, case_name in enumerate(sequences[La.CASE_ID].unique())}
        sequences.replace({La.CASE_ID: replacements}, inplace=True)
        sequences = self.sort_log(sequences)
        self.gen_seqs = (sequences.rename(columns={La.RESOURCE: 'resource'}).sort_values([La.CASE_ID, 'pos_trace']))

    def clean_time_stamps(self):
        self.gen_seqs.drop(columns=La.START_TIME)
