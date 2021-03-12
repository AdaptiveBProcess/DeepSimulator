# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import os
import copy
import shutil

import pandas as pd
from operator import itemgetter

import utils.support as sup
from utils.support import timeit, safe_exec
import readers.log_reader as lr
import readers.bpmn_reader as br
import readers.process_structure as gph
import readers.log_splitter as ls
import analyzers.sim_evaluator as sim


from core_modules.instances_generator import instances_generator as gen
from core_modules.sequences_generator import seq_generator as sg
from core_modules.times_allocator import times_generator as ta



class DeepSimulator():
    """
    Main class of the Simulator
    """
    def __init__(self, parms):
        """constructor"""
        self.parms = parms
        self.is_safe = True
        self.sim_values = list()

    def execute_pipeline(self) -> None:
        exec_times = dict()
        self.is_safe = self._read_inputs(
            log_time=exec_times, is_safe=self.is_safe)
        # modify number of instances in the model
        num_inst = len(self.log_test.caseid.unique())
        # get minimum date
        start_time = (self.log_test
                      .start_timestamp
                      .min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        print('############ Structure optimization ############')
        # Structure optimization
        seq_gen = sg.SeqGenerator({**self.parms['gl'],
                                   **self.parms['s_gen']},
                                  self.log_train)
        print('############ Generate interarrivals ############')
        self.is_safe = self._read_bpmn(
            log_time=exec_times, is_safe=self.is_safe)
        generator = gen.InstancesGenerator(self.process_graph,
                                           self.log_train,
                                           self.parms['i_gen']['gen_method'],
                                           {**self.parms['gl'],
                                            **self.parms['i_gen']})
        print('########### Generate instances times ###########')
        times_allocator = ta.TimesGenerator(self.process_graph,
                                            self.log_train,
                                            {**self.parms['gl'],
                                            **self.parms['t_gen']})
        output_path = os.path.join('output_files', sup.folder_id())
        for rep_num in range(0, self.parms['gl']['exp_reps']):
            seq_gen.generate(num_inst, start_time)
            iarr = generator.generate(num_inst, start_time)
            event_log = times_allocator.generate(seq_gen.gen_seqs, iarr)
            event_log = pd.DataFrame(event_log)
            # Export log
            self._export_log(event_log, output_path, rep_num)
            # Evaluate log
            if self.parms['gl']['evaluate']:
                self.sim_values.extend(
                    self._evaluate_logs(self.parms, self.log_test,
                                        event_log, rep_num))
        self._export_results(output_path)
        print("-- End of trial --")


    @timeit
    @safe_exec
    def _read_inputs(self, **kwargs) -> None:
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.parms['gl']['event_logs_path'],
                                             self.parms['gl']['file']),
                                self.parms['gl']['read_options'])
        # Time splitting 80-20
        self._split_timeline(0.8,
                            self.parms['gl']['read_options']['one_timestamp'])

    @timeit
    @safe_exec
    def _read_bpmn(self, **kwargs) -> None:
        bpmn_path = os.path.join(self.parms['gl']['bpmn_models'],
                                 self.parms['gl']['file'].split('.')[0]+'.bpmn')
        self.bpmn = br.BpmnReader(bpmn_path)
        self.process_graph = gph.create_process_structure(self.bpmn)

    @staticmethod
    def _evaluate_logs(parms, log, sim_log, rep_num):
        """Reads the simulation results stats
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        # print('Reading repetition:', (rep+1), sep=' ')
        sim_values = list()
        log = copy.deepcopy(log)
        log = log[~log.task.isin(['Start', 'End'])]
        log['source'] = 'log'
        log.rename(columns={'user': 'resource'}, inplace=True)
        log['caseid'] = log['caseid'].astype(str)
        log['caseid'] = 'Case' + log['caseid']
        evaluator = sim.SimilarityEvaluator(
            log,
            sim_log,
            parms['gl'],
            max_cases=1000)
        metrics = [parms['gl']['sim_metric']]
        if 'add_metrics' in parms['gl'].keys():
            metrics = list(set(list(parms['gl']['add_metrics']) + metrics))
        for metric in metrics:
            evaluator.measure_distance(metric)
            sim_values.append({**{'run_num': rep_num}, **evaluator.similarity})
        return sim_values
    
    def _export_log(self, event_log, output_path, r_num) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        event_log.to_csv(
            os.path.join(
                output_path, 'gen_'+ 
                self.parms['gl']['file'].split('.')[0]+'_'+str(r_num+1)+'.csv'), 
            index=False)

    def _export_results(self, output_path) -> None:
        # Save results
        pd.DataFrame(self.sim_values).to_csv(
            os.path.join(output_path, sup.file_id(prefix='SE_')), 
            index=False)
        # Save logs        
        log_test = self.log_test[~self.log_test.task.isin(['Start', 'End'])]
        log_test.to_csv(
            os.path.join(output_path, 'tst_'+
                         self.parms['gl']['file'].split('.')[0]+'.csv'), 
            index=False)
        if self.parms['gl']['save_models']:
            paths = ['bpmn_models', 'embedded_path', 'ia_gen_path', 
                     'seq_flow_gen_path', 'times_gen_path']
            sources = list()
            for path in paths:
                for root, dirs, files in os.walk(self.parms['gl'][path]):
                    for file in files:
                        if self.parms['gl']['file'].split('.')[0] in file:
                            sources.append(os.path.join(root, file))
            for source in sources:
                base_folder = os.path.join(
                    output_path, os.path.basename(os.path.dirname(source)))
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)
                destination = os.path.join(base_folder, 
                                           os.path.basename(source))
                shutil.copyfile(source, destination)

    @staticmethod
    def _save_times(times, parms):
        times = [{**{'output': parms['output']}, **times}]
        log_file = os.path.join('output_files', 'execution_times.csv')
        if not os.path.exists(log_file):
                open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)

# =============================================================================
# Support methods
# =============================================================================
    def _split_timeline(self, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        # Split log data
        splitter = ls.LogSplitter(self.log.data)
        train, test = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(test) < int(total_events*0.1):
            train, test = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        test = pd.DataFrame(test)
        train = pd.DataFrame(train)
        self.log_test = (test.sort_values(key, ascending=True)
                         .reset_index(drop=True))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True)
                                .reset_index(drop=True).to_dict('records'))

    @staticmethod
    def _get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(
                list(filter(lambda x: (x['caseid'] == case), data)),
                key=itemgetter(order_key))
            traces.append(trace)
        return traces
