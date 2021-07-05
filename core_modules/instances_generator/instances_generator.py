# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:51:24 2020

@author: Manuel Camargo
"""

import itertools

import pandas as pd

import readers.log_splitter as ls
from core_modules.instances_generator import dl_generators as dl
from core_modules.instances_generator import pdf_generators as pdf
from core_modules.instances_generator import prophet_generator as prf
from core_modules.instances_generator import multi_pdf_generators as mpdf


class InstancesGenerator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, process_graph, log, method, parms):
        """constructor"""
        self.log = log
        self._split_timeline(0.8, parms['read_options']['one_timestamp'])
        self.tasks = self._analize_first_tasks(process_graph)
        self.one_timestamp = parms['read_options']['one_timestamp']
        self.timeformat = parms['read_options']['timeformat']

        self.ia_times = self._mine_interarrival_time(self.log_train,
                                                     self.tasks,
                                                     self.one_timestamp)
        self.ia_valdn = self._mine_interarrival_time(self.log_valdn,
                                                     self.tasks,
                                                     self.one_timestamp)
        self.parms = parms
        self._get_generator(method)


    def generate(self, num_instances, start_time):
        return self.generator.generate(num_instances, start_time)


    def _get_generator(self, method):
        if method == 'pdf':
            self.generator = pdf.PDFGenerator(self.ia_times,
                                              self.ia_valdn)
        elif method == 'dl':
            self.generator = dl.DeepLearningGenerator(self.ia_times,
                                                      self.ia_valdn,
                                                      self.parms)
        elif method == 'mul_pdf':
            self.generator = mpdf.MultiPDFGenerator(self.ia_times,
                                                    self.ia_valdn,
                                                    self.parms)
        elif method == 'prophet':
            self.generator = prf.ProphetGenerator(self.log,
                                                  self.log_valdn,
                                                  self.parms)
        elif method == 'test':
            self.generator = self.OriginalInterarrival()
        else:
            raise ValueError('Unexistent generator')


    class OriginalInterarrival():
        
        def generate(self, log, start_time):
            i_arr = log.groupby('caseid').start_timestamp.min().reset_index()
            i_arr.rename(columns={'start_timestamp': 'timestamp'}, inplace=True)
            i_arr.drop(columns='caseid', inplace=True)
            i_arr.sort_values('timestamp', inplace=True)
            i_arr['caseid'] = i_arr.index + 1
            i_arr['caseid'] = i_arr['caseid'].astype(str)
            i_arr['caseid'] = 'Case' + i_arr['caseid']
            return i_arr

# =============================================================================
# Support modules
# =============================================================================

    @staticmethod
    def _mine_interarrival_time(log_train, tasks, one_ts):
        """
        Extracts the interarrival distribution from data

        Returns
        -------
        inter_arrival_times : list

        """
        # Analysis of start tasks
        ordering_field = 'end_timestamp' if one_ts else 'start_timestamp'
        # Find the initial activity
        log_train = log_train[log_train.task.isin(tasks)]
        arrival_timestamps = (pd.DataFrame(
            log_train.groupby('caseid')[ordering_field].min())
            .reset_index()
            .rename(columns={ordering_field:'timestamp'}))
        # group by day and calculate inter-arrival
        inter_arrival_times = list()
        # for key, group in arrival_timestamps.groupby('date'):
        daily_times = arrival_timestamps.sort_values('timestamp').to_dict('records')
        for i, event in enumerate(daily_times):
            delta = (
                daily_times[i]['timestamp'] -
                daily_times[i-1]['timestamp']).total_seconds() if i > 0 else 0
            time = daily_times[i]['timestamp'].time()
            time = time.second + time.minute*60 + time.hour*3600
            inter_arrival_times.append(
                {'caseid': daily_times[i]['caseid'],
                 'inter_time': delta,
                 'timestamp': daily_times[i]['timestamp'],
                 'daytime': time,
                 'weekday': daily_times[i]['timestamp'].weekday()})
        return pd.DataFrame(inter_arrival_times)


    def _analize_first_tasks(self, process_graph) -> list():
        """
        Extracts the first tasks of the process

        Parameters
        ----------
        process_graph : Networkx di-graph

        Returns
        -------
        list of tasks

        """
        temp_process_graph = process_graph.copy()
        for node in list(temp_process_graph.nodes):
            if process_graph.nodes[node]['type'] not in ['start', 'end', 'task']:
                preds = list(temp_process_graph.predecessors(node))
                succs = list(temp_process_graph.successors(node))
                temp_process_graph.add_edges_from(
                    list(itertools.product(preds, succs)))
                temp_process_graph.remove_node(node)
        graph_data = (pd.DataFrame.from_dict(
            dict(temp_process_graph.nodes.data()), orient='index'))
        start = graph_data[graph_data.type.isin(['start'])]
        start = start.index.tolist()[0]  # start node id
        in_tasks = [temp_process_graph.nodes[x]['name']
                    for x in temp_process_graph.successors(start)]
        return in_tasks

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
        train, valdn = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log.data)
        # Check size and change time splitting method if necesary
        if len(valdn) < int(total_events*0.1):
            train, valdn = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)
        self.log_valdn = (valdn.sort_values(key, ascending=True)
                          .reset_index(drop=True))
        self.log_train = (train.sort_values(key, ascending=True)
                          .reset_index(drop=True))
        #--
        # self.log_train.to_csv('log_train_manual.csv')
        # self.log_valdn.to_csv('log_valdn_manual.csv')
