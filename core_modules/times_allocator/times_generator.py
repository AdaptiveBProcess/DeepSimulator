# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:51:24 2020

@author: Manuel Camargo
"""
import os
import itertools
import json
import shutil
from datetime import datetime, timedelta
import utils.support as sup
from tqdm import tqdm
import math

from operator import itemgetter
import numpy as np
import pandas as pd
import traceback
import time
import multiprocessing
from multiprocessing import Pool


import readers.log_splitter as ls
from extraction import log_replayer as rpl
from core_modules.times_allocator import embedding_trainer as em
from core_modules.times_allocator import times_model_optimizer as to
from core_modules.times_allocator import model_hpc_optimizer as hpc_op


from sklearn.preprocessing import MaxAbsScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from pickle import dump, load
import keras.utils as ku


class TimesGenerator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, process_graph, log, parms):
        """constructor"""
        self.log = log
        self.process_graph = process_graph
        self.parms = parms

        self.one_timestamp = parms['read_options']['one_timestamp']
        self.timeformat = parms['read_options']['timeformat']
        self.model_metadata = dict()
        self._load_model()

# =============================================================================
# Generate traces
# =============================================================================
    def _load_model(self) -> None:
        model_path = os.path.join(
            self.parms['times_gen_path'],
            self.parms['file'].split('.')[0]+'.h5')
        if os.path.exists(model_path) and not self.parms['update_times_gen']:
            self.model_path = model_path
        elif os.path.exists(model_path) and self.parms['update_times_gen']:
            self.model_path = model_path
            self._discover_model(False)
        elif not os.path.exists(model_path):
            self.model_path = model_path
            self._discover_model(False)

    def generate(self, sequences, iarr):
        return self._generate_traces_parallel(sequences, iarr)

# =============================================================================
# Train model
# =============================================================================

    def _discover_model(self, compare, **kwargs):
        # indexes creation
        self._indexing()
        # replay
        self._replay_process()
        self._split_timeline(0.8, self.one_timestamp)
        
        self.log_train = self._add_calculated_times(self.log_train)
        self.log_valdn = self._add_calculated_times(self.log_valdn)
        # Add index to the event log
        ac_idx = lambda x: self.ac_index[x['task']]
        self.log_train['ac_index'] = self.log_train.apply(ac_idx, axis=1)
        self.log_valdn['ac_index'] = self.log_valdn.apply(ac_idx, axis=1)
        # Load embedding matrixes
        emb_trainer = em.EmbeddingTrainer(self.parms,
                                          pd.DataFrame(self.log),
                                          self.ac_index, 
                                          self.index_ac)
        self.ac_weights = emb_trainer.load_embbedings()
        # Scale features
        self._transform_features()
        # Optimizer
        self.parms['output'] = os.path.join('output_files', sup.folder_id())
        if self.parms['opt_method'] == 'rand_hpc':
            times_optimizer = hpc_op.ModelHPCOptimizer(self.parms,
                                                       self.log_train, 
                                                       self.log_valdn,
                                                       self.ac_index,
                                                       self.ac_weights)
            times_optimizer.execute_trials()
        elif self.parms['opt_method'] == 'bayesian':
            times_optimizer = to.TimesModelOptimizer(self.parms, 
                                                     self.log_train, 
                                                     self.log_valdn,
                                                     self.ac_index,
                                                     self.ac_weights)
            times_optimizer.execute_trials()
        acc = times_optimizer.best_loss

        metadata_file = os.path.join(
            self.parms['times_gen_path'],
            self.parms['file'].split('.')[0]+'_meta.json')
        # compare with existing model
        save = True
        if compare:
            # Loading of parameters from existing model
            if os.path.exists(metadata_file):
                with open(metadata_file) as file:
                    data = json.load(file)
                if data['loss'] < acc:
                    save = False
                    print('dont save')
        if save:
            # best structure mining parameters
            self.model_metadata['loss'] = acc
            self.model_metadata['generated_at'] = (
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            self.model_metadata['ac_index'] = self.ac_index
            self.model_metadata = {**self.model_metadata, 
                                   **times_optimizer.best_parms}
            print(self.model_metadata)           
            # Copy best model to destination folder
            source = os.path.join(times_optimizer.best_output,
                             self.parms['file'].split('.')[0]+'.h5')

            shutil.copyfile(source, self.model_path)
            # Save metadata
            sup.create_json(self.model_metadata, metadata_file)
        # export scaler https://bit.ly/2L2dpt1
        dump(self.scaler, 
             open(os.path.join(
                 self.parms['times_gen_path'], 
                 self.parms['file'].split('.')[0]+'_scaler.pkl'),'wb'))
        # clean output folder
        shutil.rmtree(self.parms['output'])

# =============================================================================
# generate_traces
# =============================================================================

    def _generate_traces(self, sequences, iarr):
        metadata_file = os.path.join(
            self.parms['times_gen_path'],
            self.parms['file'].split('.')[0]+'_meta.json')
        # Loading of parameters from existing model
        if os.path.exists(metadata_file):
            with open(metadata_file) as file:
                data = json.load(file)
                self.ac_index = data['ac_index']
                n_size = data['n_size']
        index_ac = {v:k for k, v in self.ac_index.items()}
        self.scaler = load(open(os.path.join(
                 self.parms['times_gen_path'], 
                 self.parms['file'].split('.')[0]+'_scaler.pkl'), 'rb'))
        iarr = {x['caseid']: x['timestamp'] for x in iarr.to_dict('records')}
        sequences = sequences[~sequences.task.isin(['Start', 'End'])]
        # print(sequences[sequences.caseid=='Case1'])
        event_log = list()
        for cid, trace in tqdm(sequences
                               .sort_values('pos_trace')
                               .groupby('caseid'), desc='generating traces:'):
            s_timestamp = iarr[cid]
            proc_t, wait_t = 0.0, 0.0
            act_ngram = [0 for i in range(n_size)]
            feat_ngram = np.zeros((1, n_size, 10))
            for event in trace.to_dict('records'):
                new_event = dict()
                daytime = s_timestamp.time()
                daytime = daytime.second + daytime.minute*60 + daytime.hour*3600
                daytime = daytime / 86400
                day_dummies = ku.to_categorical(s_timestamp.weekday(), 
                                                num_classes=7)
                record = list(day_dummies) + [proc_t, wait_t, daytime]
                feat_ngram = np.append(feat_ngram, [[record]], axis=1)
                feat_ngram = np.delete(feat_ngram, 0, 1)
                act_ngram.append(self.ac_index[event['task']])
                act_ngram.pop(0)
                preds = self.model.predict({'ac_input': np.array([act_ngram]), 
                                            'features': feat_ngram})
                ipred = self.scaler.inverse_transform(preds)
                proc_t, wait_t = preds[0]
                new_event['caseid'] = cid
                new_event['resource'] = event['resource']
                new_event['task'] = (index_ac[event['task']] 
                                     if type(event['task']) is int 
                                     else event['task'])
                new_event['start_timestamp'] = s_timestamp
                new_event['end_timestamp'] = (
                    s_timestamp + timedelta(seconds=float(ipred[0][0])))
                s_timestamp = (
                    new_event['end_timestamp'] + timedelta(
                        seconds=float(ipred[0][1])))
                event_log.append(new_event)
        return event_log
                
    def _generate_traces_parallel(self, sequences, iarr):
        tf.compat.v1.reset_default_graph()
        metadata_file = os.path.join(
            self.parms['times_gen_path'],
            self.parms['file'].split('.')[0]+'_meta.json')
        # Loading of parameters from existing model
        if os.path.exists(metadata_file):
            with open(metadata_file) as file:
                data = json.load(file)
                self.ac_index = data['ac_index']
                n_size = data['n_size']
        s_path = os.path.join(
                 self.parms['times_gen_path'], 
                 self.parms['file'].split('.')[0]+'_scaler.pkl')
        iarr = {x['caseid']: x['timestamp'] for x in iarr.to_dict('records')}
        sequences = sequences[~sequences.task.isin(['Start', 'End'])]
        def pbar_async(p, msg):
            pbar = tqdm(total=reps, desc=msg)
            processed = 0
            while not p.ready():
                cprocesed = (reps - p._number_left)
                if processed < cprocesed:
                    increment = cprocesed - processed
                    pbar.update(n=increment)
                    processed = cprocesed
            time.sleep(1)
            pbar.update(n=(reps - processed))
            p.wait()
            pbar.close()

        cpu_count = multiprocessing.cpu_count()
        cases = list(sequences.caseid.unique())
        b_size = math.ceil(len(cases)/(cpu_count*2))
        chunks = [cases[x:x+b_size] for x in range(0, len(cases), b_size)]
        reps = len(chunks)
        pool = Pool(processes=cpu_count)
        # Generate
        args = [(iarr, 
                 sequences[sequences.caseid.isin(cases)], 
                 n_size, 
                 self.ac_index, 
                 self.model_path, 
                 s_path) for cases in chunks]
        p = pool.map_async(self.generate_trace, args)
        pbar_async(p, 'generating traces:')
        pool.close()
        # Save results
        event_log = list(itertools.chain(*p.get()))
        return event_log

    @staticmethod
    def generate_trace(args):
        def gen(iarr, batch, n_size, ac_index, model_path, s_path):
            """Reads the simulation results stats
            Args:
                settings (dict): Path to jar and file names
                rep (int): repetition number
            """
            try:
                index_ac = {v:k for k, v in ac_index.items()}
                model = load_model(model_path)
                new_batch = list()
                for cid, trace in batch.sort_values('pos_trace').groupby('caseid'):
                    s_timestamp = iarr[cid]
                    proc_t, wait_t = 0.0, 0.0
                    act_ngram = [0 for i in range(n_size)]
                    feat_ngram = np.zeros((1, n_size, 10))
                    for event in trace.to_dict('records'):
                        new_event = dict()
                        daytime = s_timestamp.time()
                        daytime = daytime.second + daytime.minute*60 + daytime.hour*3600
                        daytime = daytime / 86400
                        day_dummies = ku.to_categorical(s_timestamp.weekday(), 
                                                        num_classes=7)
                        record = list(day_dummies) + [proc_t, wait_t, daytime]
                        feat_ngram = np.append(feat_ngram, [[record]], axis=1)
                        feat_ngram = np.delete(feat_ngram, 0, 1)
                        act_ngram.append(ac_index[event['task']])
                        act_ngram.pop(0)
                        preds = model.predict({'ac_input': np.array([act_ngram]),
                                               'features': feat_ngram})
                        preds[preds < 0] = 0
                        scaler = load(open(s_path, 'rb'))
                        ipred = scaler.inverse_transform(preds)
                        proc_t, wait_t = preds[0]
                        new_event['caseid'] = cid
                        new_event['resource'] = event['resource']
                        new_event['task'] = (index_ac[event['task']] 
                                             if type(event['task']) is int 
                                             else event['task'])
                        new_event['start_timestamp'] = s_timestamp
                        new_event['end_timestamp'] = (
                            s_timestamp + timedelta(seconds=float(ipred[0][0])))
                        s_timestamp = (
                            new_event['end_timestamp'] + timedelta(
                                seconds=float(ipred[0][1])))
                        new_batch.append(new_event)
                return new_batch
            except Exception:
                traceback.print_exc()
        return gen(*args)
        
# =============================================================================
# Support modules
# =============================================================================
    def _replay_process(self) -> None:
        """
        Process replaying
        """
        replayer = rpl.LogReplayer(self.process_graph, 
                                   self.log.get_traces(),
                                   self.parms, 
                                   msg='reading conformant training traces:')
        self.log = (replayer.process_stats
                    .rename(columns={'resource':'user'})
                    .to_dict('records'))
    

    def _indexing(self):
        # Activities index creation
        log = pd.DataFrame(self.log.data)
        log = log[~log.task.isin(['Start', 'End'])]
        subsec_set = log.task.unique().tolist()
        self.ac_index = dict()
        for i, _ in enumerate(subsec_set):
            self.ac_index[subsec_set[i]] = i + 1
        self.ac_index['Start'] = 0
        self.ac_index['End'] = len(self.ac_index)
        self.index_ac = {v: k for k, v in self.ac_index.items()}

    
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
        splitter = ls.LogSplitter(self.log)
        train, valdn = splitter.split_log('timeline_contained', size, one_ts)
        total_events = len(self.log)
        # Check size and change time splitting method if necesary
        if len(valdn) < int(total_events*0.1):
            train, valdn = splitter.split_log('timeline_trace', size, one_ts)
        # Set splits
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        valdn = pd.DataFrame(valdn)
        train = pd.DataFrame(train)
        valdn = valdn[~valdn.task.isin(['Start', 'End'])]
        train = train[~train.task.isin(['Start', 'End'])]
        self.log_valdn = (valdn.sort_values(key, ascending=True)
                          .reset_index(drop=True))
        self.log_train = (train.sort_values(key, ascending=True)
                          .reset_index(drop=True))

    def _add_calculated_times(self, log):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log['daytime'] = 0
        log = log.to_dict('records')
        log = sorted(log, key=lambda x: x['caseid'])
        for _, group in itertools.groupby(log, key=lambda x: x['caseid']):
            events = list(group)
            events = sorted(events, key=itemgetter('start_timestamp'))
            for i in range(0, len(events)):
                time = events[i]['start_timestamp'].time()
                time = time.second + time.minute*60 + time.hour*3600
                events[i]['daytime'] = time
                events[i]['weekday'] = events[i]['start_timestamp'].weekday()
        return pd.DataFrame.from_dict(log)

    def _transform_features(self):
        # scale continue features
        cols = ['processing_time', 'waiting_time']
        self.scaler = MaxAbsScaler()
        self.scaler.fit(self.log_train[cols])
        self.log_train[cols] = self.scaler.transform(self.log_train[cols])
        self.log_valdn[cols] = self.scaler.transform(self.log_valdn[cols])
        # scale daytime
        self.log_train['daytime'] = np.divide(self.log_train['daytime'], 86400)
        self.log_valdn['daytime'] = np.divide(self.log_valdn['daytime'], 86400)
        # filter data
        cols.extend(['caseid', 'ac_index', 'daytime', 'weekday'])
        self.log_train = self.log_train[cols]
        self.log_valdn = self.log_valdn[cols]
