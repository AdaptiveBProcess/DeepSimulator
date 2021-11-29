# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:45:56 2021

@author: Manuel Camargo
"""
import os
import json
import math
import itertools
from tqdm import tqdm
import time
from datetime import timedelta

import numpy as np

import traceback
import multiprocessing
from multiprocessing import Pool

import tensorflow as tf
from tensorflow.keras.models import load_model
from pickle import load
from keras.utils import np_utils as ku


class NoIntercasesPredictor():
    """
    """
    def __init__(self, model_path, parms):
        """constructor"""
        self.model_path = model_path
        self.parms = parms


    def predict(self, sequences, iarr):
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
