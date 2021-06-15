# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:14:33 2020

@author: Manuel Camargo
"""
# #%%
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing
from multiprocessing import Pool
from math import ceil

from datetime import datetime, timedelta

import json 
import scipy.stats as st

import utils.support as sup
import analyzers.sim_evaluator as sim

import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import time
import traceback


##%%
class MultiPDFGenerator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, ia_times, ia_valdn, parms):
        """constructor"""
        self.ia_times = ia_times
        self.ia_valdn = ia_valdn
        self.parms = parms
        self.model_metadata = dict()
        self._load_model()

    # @safe_exec
    def _load_model(self) -> None:
        filename = os.path.join(self.parms['ia_gen_path'],
                                self.parms['file'].split('.')[0]+'_mpdf.json')
        if os.path.exists(filename) and not self.parms['update_mpdf_gen']:
            with open(filename) as file:
                self.model = json.load(file)
        elif os.path.exists(filename) and self.parms['update_mpdf_gen']:
            with open(filename) as file:
                self.model = json.load(file)
            self._create_model(True)
        elif not os.path.exists(filename):
            self._create_model(False)
        # self._generate_traces(num_instances, start_time)
        # self.times['caseid'] = self.times.index + 1
        # self.times['caseid'] = self.times['caseid'].astype(str)
        # self.times['caseid'] = 'Case' + self.times['caseid']
        # return self.times

    def _create_model(self, compare):
        # hours = [8]
        hours = [1, 2, 4, 8, 12]
        args = [(w, self.ia_times, self.ia_valdn, self.parms) for w in hours]
        reps = len(args)
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
        w_count =  reps if reps <= cpu_count else cpu_count
        pool = Pool(processes=w_count)
        # Simulate
        p = pool.map_async(self.create_evaluate_model, args)
        pbar_async(p, 'evaluating models:')
        pool.close()
        # Save results
        element = min(p.get(), key=lambda x: x['loss'])
        metadata_file = os.path.join(
            self.parms['ia_gen_path'],
            self.parms['file'].split('.')[0]+'_mpdf_meta.json')
        # compare with existing model
        save = True
        if compare:
            # Loading of parameters from existing model
            if os.path.exists(metadata_file):
                with open(metadata_file) as file:
                    data = json.load(file)
                    data = {k: v for k, v in data.items()}
                if data['loss'] < element['loss']:
                    save = False
                    print('dont save')
        if save:
            self.model = element['model']
            sup.create_json(self.model, os.path.join(
            self.parms['ia_gen_path'], 
            self.parms['file'].split('.')[0]+'_mpdf.json'))
    
            # best structure mining parameters
            self.model_metadata['window'] = element['model']['window']
            self.model_metadata['loss'] = element['loss']
            self.model_metadata['generated_at'] = (
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
            sup.create_json(self.model_metadata, metadata_file)            
            
    @staticmethod
    def create_evaluate_model(args):
        
        def dist_best(data, window):
            """
            Finds the best probability distribution for a given data serie
            """
            # Create a data series from the given list
            # data = pd.Series(self.data_serie)
            # plt.hist(data, bins=self.bins, density=True, range=self.window)
            # plt.show()
            # Get histogram of original data
            hist, bin_edges = np.histogram(data, bins='auto', range=window)
            bin_edges = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0
            # Distributions to check
            distributions = [st.norm, st.expon, st.uniform,
                             st.triang, st.lognorm, st.gamma]
            # Best holders
            best_distribution = st.norm
            best_sse = np.inf
            best_loc = 0
            best_scale = 0
            best_args = 0
            # Estimate distribution parameters from data
            for distribution in distributions:
                # Try to fit the distribution
                try:
                    # Ignore warnings from data that can't be fit
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        # fit dist to data
                        params = distribution.fit(data)
                        # Separate parts of parameters
                        arg = params[:-2]
                        loc = params[-2]
                        scale = params[-1]
                        # Calculate fitted PDF and error with fit in distribution
                        pdf = distribution.pdf(bin_edges, 
                                               loc=loc, 
                                               scale=scale, 
                                               *arg)
                        sse = np.sum(np.power(hist - pdf, 2.0))
                        # identify if this distribution is better
                        if best_sse > sse > 0:
                            best_distribution = distribution
                            best_sse = sse
                            best_loc = loc
                            best_scale = scale
                            best_args = arg
                except:
                    pass
            return {'dist': best_distribution.name,
                    'loc': best_loc,
                    'scale': best_scale,
                    'args': best_args}
        
        def generate_traces(model, num_instances, start_time):
            dobj = {'norm': st.norm, 'expon': st.expon,
                    'uniform': st.uniform, 'triang': st.triang, 
                    'lognorm': st.lognorm, 'gamma': st.gamma}
            timestamp = datetime.strptime(start_time,
                                          "%Y-%m-%dT%H:%M:%S.%f+00:00")
            times = list()
            # clock = timestamp.floor(freq ='H')
            clock = (timestamp.replace(microsecond=0, second=0, minute=0)
                     - timedelta(hours=1))
            i = 0
            def add_ts(timestamp, dname):
                times.append({'dname': dname,
                              'timestamp': timestamp})
                return times
            
            # print(clock)
            while i < num_instances:
                # print('Clock:', clock)
                try:
                    window = str(model['daily_windows'][str(clock.hour)])
                    day = str(clock.weekday())
                    dist = model['distribs'][window][day]
                except KeyError:
                    dist = None
                if dist is not None:
                    missing = min((num_instances - i), dist['num'])
                    if dist['dist'] in ['norm', 'expon', 'uniform']:
                        # TODO: Check parameters 
                        gen_inter = dobj[dist['dist']].rvs(loc=dist['loc'],
                                                           scale=dist['scale'],
                                                           size=missing)
                    elif dist['dist'] == 'lognorm':
                        m = dist['mean']
                        v = dist['var']
                        phi = np.sqrt(v + m**2)
                        mu = np.log(m**2/phi)
                        sigma = np.sqrt(np.log(phi**2/m**2))
                        sigma = sigma if sigma > 0.0 else 0.000001
                        gen_inter = dobj[dist['dist']].rvs(sigma,
                                                           scale=np.exp(mu),
                                                           size=missing)
                    elif dist['dist'] in ['gamma', 'triang']:
                        gen_inter = dobj[dist['dist']].rvs(dist['args'],
                                                           loc=dist['loc'],
                                                           scale=dist['scale'],
                                                           size=missing)
                    else:
                        clock += timedelta(seconds=3600*model['window'])
                        print('Not implemented: ', dist['dist'])
                    #TODO: check the generated negative values
                    timestamp = clock
                    neg = 0
                    for inter in gen_inter:
                        if inter > 0:
                            timestamp += timedelta(seconds=inter)
                            if timestamp < clock + timedelta(seconds=3600*model['window']):
                                add_ts(timestamp, dist['dist'])
                            else:
                                neg +=1
                        else:
                            neg +=1
                    i += len(gen_inter) - neg
                    # print(neg)
                    # print(i)
                #TODO: Check if the clock is not been skipped
                try:
                    clock += timedelta(seconds=3600*model['window'])
                except:
                    print(clock)
                    print(model['window'])
                    print(3600*model['window'])
                    sys.exit(1)
            # pd.DataFrame(times).to_csv('times.csv')
            return pd.DataFrame(times)
        
        def create_model(window, ia_times, ia_valdn, parms):
            try:
                hist_range = [0, int((window * 3600))]
                day_hour = lambda x: x['timestamp'].hour
                ia_times['hour'] = ia_times.apply(day_hour, axis=1)
                date = lambda x: x['timestamp'].date()
                ia_times['date'] = ia_times.apply(date, axis=1)
                # create time windows
                i = 0
                daily_windows = dict()
                for x in range(24):
                    if x % window == 0:
                        i += 1
                    daily_windows[x] = i
                ia_times = ia_times.merge(
                    pd.DataFrame.from_dict(daily_windows, 
                                           orient='index').rename_axis('hour'),
                    on='hour',
                    how='left').rename(columns={0: 'window'})
                inter_arrival = list()
                for key, group in ia_times.groupby(['window', 'date', 'weekday']):
                    w_df = group.copy()
                    w_df = w_df.reset_index()
                    prev_time = w_df.timestamp.min().floor(freq ='H')
                    for i, item in w_df.iterrows():
                        inter_arrival.append(
                            {'window': key[0],
                             'weekday': item.weekday,
                             'intertime': (item.timestamp - prev_time).total_seconds(),
                             'date': item.date})
                        prev_time = item.timestamp
                distribs = dict()
                for key, group in pd.DataFrame(inter_arrival).groupby(['window', 'weekday']):
                    intertime = group.intertime
                    if len(intertime)>2:
                        intertime = intertime[intertime.between(
                            intertime.quantile(.15), intertime.quantile(.85))]
                    distrib = dist_best(intertime, hist_range)
                    # TODO: averiguar porque funciona con la mitad de los casos???
                    number = group.groupby('date').intertime.count()
                    if len(number)>2:
                        number = number[number.between(
                            number.quantile(.15), number.quantile(.85))]
                    # distrib['num'] = int(number.median()/2)
                    distrib['num'] = ceil(number.median()/2)
                    # distrib['num'] = int(number.median())
                    if distrib['dist'] == 'lognorm':
                        distrib['mean'] = np.mean(group.intertime)
                        distrib['var'] = np.var(group.intertime)
                    distribs[str(key[0])] = {str(key[1]): distrib}
                model = {'window': window,
                         'daily_windows': {str(k):v for k, v in daily_windows.items()},
                         'distribs': distribs}
                
                # validation
                # modify number of instances in the model
                num_inst = len(ia_valdn.caseid.unique())
                # get minimum date
                start_time = (ia_valdn
                              .timestamp
                              .min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
                times = generate_traces(model, num_inst, start_time)
                # ia_valdn = ia_valdn[['caseid', 'timestamp']]
                # times = times[['caseid', 'timestamp']]
                evaluation = sim.SimilarityEvaluator(ia_valdn, times, parms, 0, dtype='serie')
                evaluation.measure_distance('hour_emd')
                return {'model': model, 'loss': evaluation.similarity['sim_val']}
            except Exception:
                traceback.print_exc()
                return {'model': [], 'loss': 1}
        return create_model(*args)


    def generate(self, num_instances, start_time):
        dobj = {'norm': st.norm, 'expon': st.expon,
                'uniform': st.uniform, 'triang': st.triang, 
                'lognorm': st.lognorm, 'gamma': st.gamma}
        timestamp = datetime.strptime(start_time,
                                      "%Y-%m-%dT%H:%M:%S.%f+00:00")
        times = list()
        # clock = timestamp.floor(freq ='H')
        clock = (timestamp.replace(microsecond=0, second=0, minute=0)
                 - timedelta(hours=1))
        i = 0
        def add_ts(timestamp, dname):
            times.append({'dname': dname,
                          'timestamp': timestamp})
            return times
        
        # print(clock)
        while i < num_instances:
            # print('Clock:', clock)
            try:
                window = str(self.model['daily_windows'][str(clock.hour)])
                day = str(clock.weekday())
                dist = self.model['distribs'][window][day]
            except KeyError:
                dist = None
            if dist is not None:
                missing = min((num_instances - i), dist['num'])
                if dist['dist'] in ['norm', 'expon', 'uniform']:
                    # TODO: Check parameters 
                    gen_inter = dobj[dist['dist']].rvs(loc=dist['loc'],
                                                       scale=dist['scale'],
                                                       size=missing)
                elif dist['dist'] == 'lognorm':
                    m = dist['mean']
                    v = dist['var']
                    phi = np.sqrt(v + m**2)
                    mu = np.log(m**2/phi)
                    sigma = np.sqrt(np.log(phi**2/m**2))
                    sigma = sigma if sigma > 0.0 else 0.000001
                    gen_inter = dobj[dist['dist']].rvs(sigma,
                                                       scale=np.exp(mu),
                                                       size=missing)
                elif dist['dist'] in ['gamma', 'triang']:
                    gen_inter = dobj[dist['dist']].rvs(dist['args'],
                                                       loc=dist['loc'],
                                                       scale=dist['scale'],
                                                       size=missing)
                else:
                    clock += timedelta(seconds=3600*self.model['window'])
                    print('Not implemented: ', dist['dist'])
                #TODO: check the generated negative values
                timestamp = clock
                neg = 0
                for inter in gen_inter:
                    if inter > 0:
                        timestamp += timedelta(seconds=inter)
                        if timestamp < clock + timedelta(seconds=3600*self.model['window']):
                            add_ts(timestamp, dist['dist'])
                        else:
                            neg +=1
                    else:
                        neg +=1
                i += len(gen_inter) - neg
            #TODO: Check if the clock is not been skipped
            clock += timedelta(seconds=3600*self.model['window'])
        # pd.DataFrame(times).to_csv('times.csv')
        self.times = pd.DataFrame(times)
        self.times['caseid'] = self.times.index + 1
        self.times['caseid'] = self.times['caseid'].astype(str)
        self.times['caseid'] = 'Case' + self.times['caseid']
        return self.times


    @staticmethod
    def _graph_timeline(log) -> None:
        time_series = log.copy()[['caseid', 'timestamp']]
        time_series['occ'] = 1
        time_series.set_index('timestamp', inplace=True)
        time_series.occ.rolling('3h').sum().plot(figsize=(30,10), linewidth=5, fontsize=10)
        plt.xlabel('Days', fontsize=20);
        print(time_series)
