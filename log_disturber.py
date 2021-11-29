# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:55:36 2021

@author: Manuel Camargo
"""
import os
import sys
import getopt
# import shutil
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import readers.log_reader as lr
from support_modules.writers import xml_writer_alt as xes

import itertools
from operator import itemgetter


class LogDisturber():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, parms):
        self.input_log = os.path.join(parms['gl']['event_logs_path'],
                                      parms['gl']['file'])
        self.output_log = os.path.join(parms['gl']['event_logs_path'],
                                       (parms['gl']['file']
                                        .split('.')[0] + '_disturbed'))

        self.read_options = parms['gl']['read_options']

    def disturb_log(self, method):
        self._read_file()
        self._extract_case_starts()
        disturber = self._get_disturber(method)
        disturber()
        self.create_graph(self.log, self.new_log)
        # self.export_disturbed_log()

    def _get_disturber(self, method):
        if method == 'reduce':
            return self._reduce_arrivals
        elif method == 'reduce_times':
            return self._reduce_arrivals_times
        else:
            raise ValueError(method)

    def _read_file(self):
        log = lr.LogReader(self.input_log, self.read_options)
        self.log = pd.DataFrame(log.data)

    def _extract_case_starts(self):
        df_starts = self.log.groupby('caseid').start_timestamp.min().reset_index()
        self.df_starts = df_starts.sort_values('start_timestamp').reset_index(drop=True)

    @staticmethod
    def create_graph(log, new_log):
        df_join = list()
        for source, df in [('original', log), ('modified', new_log)]:
            df_starts = df.groupby('caseid').start_timestamp.min().reset_index()
            df_starts = df_starts.sort_values('start_timestamp').reset_index(drop=True)
            df_train = df_starts.groupby(
                [pd.Grouper(key='start_timestamp', freq='H')]).size().reset_index(name='count')
            df_train.rename(columns={'start_timestamp': 'ds', 'count': 'y'}, inplace=True)
            df_train = df_train.fillna(1)
            df_train['source'] = source
            if source == 'original':
                df_train['chunk'] = 0
            df_join.append(df_train)
        df_join = pd.concat(df_join, axis=0).reset_index(drop=True)
        sns.lineplot(data=df_join, x="ds", y="y", hue='source')
        plt.show()

    def _reduce_arrivals(self):
        new_evlog = list()
        chunks_data = list()
        df_list = np.array_split(self.df_starts, 6)
        for i, cases in enumerate(df_list):
            if i % 2 == 0:
                print(i, 'pares')
                cases['chunk'] = i
                new_evlog.append(cases)
            else:
                print(i, 'impares')
                cases['chunk'] = i
                new_evlog.append(cases.iloc[::3])
            chunks_data.append({'chunk': i,
                                'start_date': cases['start_timestamp'].min(),
                                'end_date': cases['start_timestamp'].max()})
        new_arrival = pd.concat(new_evlog)
        self.chunks_data = chunks_data
        self.new_log = self.log[self.log.caseid.isin(new_arrival.caseid.unique())]


    def _reduce_arrivals_times(self, reduction=0.3):
        new_evlog = list()
        chunks_data = list()
        df_list = np.array_split(self.df_starts, 6)
        for i, cases in enumerate(df_list):
            if i % 2 == 0:
                cases['chunk'] = i
                new_evlog.append(cases)
            else:
                cases['chunk'] = i
                new_evlog.append(cases.iloc[::3])
            chunks_data.append({'chunk': i,
                                'start_date': cases['start_timestamp'].min(),
                                'end_date': cases['start_timestamp'].max()})
        new_arrival = pd.concat(new_evlog)
        case_chunk = new_arrival[['caseid', 'chunk']].drop_duplicates()
        self.chunks_data = pd.DataFrame(chunks_data)
        new_log = self.log[self.log.caseid.isin(new_arrival.caseid.unique())]
        new_log = new_log.merge(case_chunk, how='left', on='caseid')
        for key, data in new_log.groupby('chunk'):
            if key % 2 != 0:
                print(key)
                new_data = self._add_calculated_times(data)
                recalculate = lambda x: (x.wait * (1 - reduction) if x.wait > 0 else 0)
                new_data['new_wait'] = new_data.apply(recalculate, axis=1)
                print(new_data)

    @staticmethod
    def _add_calculated_times(log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['dur'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('start_timestamp'))
            for i in range(0, len(events)):
                dur = (events[i]['end_timestamp'] -
                       events[i]['start_timestamp']).total_seconds()
                if i == 0:
                    wit = 0
                else:
                    wit = (events[i]['start_timestamp'] -
                           events[i-1]['end_timestamp']).total_seconds()
                events[i]['wait'] = wit if wit >= 0 else 0
                events[i]['dur'] = dur
        return pd.DataFrame.from_dict(log)


    def export_disturbed_log(self):
        xes.XesWriter(self.new_log.to_dict('records'),
                      self.read_options,
                      self.output_log + '.xes')
        self.new_log.rename(columns={'task': 'Activity', 'user': 'Resource'}, inplace=True)
        self.new_log.to_csv(self.output_log + '.csv', index=False)


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
        parms['gl']['file'] = 'BPI_Challenge_2017_W_Two_TS.xes'
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "h:f:", ['file='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parms['gl'][key] = None
                else:
                    parms['gl'][key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    log_disturber = LogDisturber(parms)
    log_disturber.disturb_log('reduce_times')


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
    parms['gl']['event_logs_path'] = os.path.join('input_files', 'event_logs')
    return parms


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])
