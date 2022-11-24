# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:51:24 2020

@author: Manuel Camargo
"""
import os
import itertools
import json
import shutil
from datetime import datetime
import utils.support as sup

from operator import itemgetter
import numpy as np
import pandas as pd

import readers.log_splitter as ls
from extraction import log_replayer as rpl
from extraction import role_discovery as rl
from core_modules.times_allocator import embedder as emb
from core_modules.times_allocator import embedding_trainer as em
from core_modules.times_allocator import times_model_optimizer as to
from core_modules.times_allocator import model_hpc_optimizer as hpc_op
from core_modules.times_allocator import intercase_features_calculator as it
from core_modules.times_allocator import times_predictor as tp
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions

from sklearn.preprocessing import MaxAbsScaler
from pickle import dump


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
        model_exist = True
        model_path = self._define_model_path(self.parms)
        # Save path(s) if the model exists 
        if isinstance(model_path, tuple):
            self.proc_model_path = model_path[0]
            self.wait_model_path = model_path[1]
            model_exist = (
                os.path.exists(model_path[0]) and os.path.exists(model_path[1]))
            self.parms['proc_model_path'] = model_path[0]
            self.parms['wait_model_path'] = model_path[1]
        else:
            self.model_path = model_path
            model_exist = os.path.exists(model_path)
            self.parms['model_path'] = model_path
        # Discover and compare
        if not model_exist or self.parms['update_times_gen']:
            times_optimizer = self._discover_model()
            save, metadata_file = self._compare_models(
                times_optimizer.best_loss, 
                self.parms['update_times_gen'], model_path)
            if save:
                self._save_model(metadata_file, times_optimizer, model_path)
            # Save basic features scaler
            name = metadata_file.replace('_meta.json', '')
            dump(self.scaler, open(name+'_scaler.pkl','wb'))
            # clean output folder
            shutil.rmtree(self.parms['output'])

            
    def generate(self, sequences, iarr):
        model_path = (self.model_path
                      if self.parms['model_type'] in ['basic', 'inter', 'inter_nt']
                      else (self.proc_model_path, self.wait_model_path))
        predictor = tp.TimesPredictor(model_path, self.parms, sequences, iarr)
        return predictor.predict(self.parms['model_type'])


    @staticmethod
    def _define_model_path(parms):
        path = parms['times_gen_path']
        fname = parms['file'].split('.')[0]
        inter = parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']
        is_dual = parms['model_type'] == 'dual_inter'
        arpool = parms['all_r_pool']
        next_ac = parms['model_type'] == 'inter_nt'
        if inter:
            if is_dual:
                if arpool:
                    return (os.path.join(path, fname+'_dpiapr.h5'),
                            os.path.join(path, fname+'_dwiapr.h5'))
                else:
                    return (os.path.join(path, fname+'_dpispr.h5'),
                            os.path.join(path, fname+'_dwispr.h5'))
            else:
                if next_ac:
                    if arpool:
                        return os.path.join(path, fname+'_inapr.h5')
                    else:
                        return os.path.join(path, fname+'_inspr.h5')
                else:
                    if arpool:
                        return os.path.join(path, fname+'_iapr.h5')
                    else:
                        return os.path.join(path, fname+'_ispr.h5')
        else:
            return os.path.join(path, fname+'.h5')

        
    def _compare_models(self, acc, model_exist, file):
        if isinstance(file, tuple):
            model = os.path.splitext(os.path.split(file[0])[1])[0]
            model = (model.replace('dpiapr', 'diapr') 
                     if self.parms['all_r_pool'] else 
                     model.replace('dpispr', 'dispr'))
            metadata_file = os.path.join(self.parms['times_gen_path'],
                                         model+'_meta.json')
        else:
            model = os.path.splitext(os.path.split(file)[1])[0]
            metadata_file = os.path.join(self.parms['times_gen_path'],
                                         model+'_meta.json')
        # compare with existing model
        save = True
        if model_exist:
            # Loading of parameters from existing model
            if os.path.exists(metadata_file):
                with open(metadata_file) as file:
                    data = json.load(file)
                if data['loss'] < acc:
                    save = False
        return save, metadata_file

    
    def _save_model(self, metadata_file, times_optimizer, model_path):
        model_metadata = dict()
        # best structure mining parameters
        model_metadata['loss'] = times_optimizer.best_loss
        model_metadata['generated_at'] = (
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        model_metadata['ac_index'] = self.ac_index
        model_metadata['usr_index'] = self.usr_index
        model_metadata['log_size'] = len(pd.DataFrame(self.log).caseid.unique())
        model_metadata = {**model_metadata, 
                          **times_optimizer.best_parms}
        model_name = metadata_file.replace('_meta.json', '')
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            model_metadata['roles'] = self.roles
            model_metadata['roles_table'] = self.roles_table.to_dict('records')
            model_metadata['inter_mean_states'] = self.mean_states
            # Save intecase scalers
            dump(self.inter_scaler, open(model_name+'_inter_scaler.pkl', 'wb'))
            if self.parms['model_type'] == 'dual_inter':
                dump(self.end_inter_scaler,
                     open(model_name+'_end_inter_scaler.pkl', 'wb'))
        # Save models
        if isinstance(model_path, tuple):
            shutil.copyfile(os.path.join(times_optimizer.best_output,
                                         os.path.split(model_path[0])[1]),
                            self.proc_model_path)
            shutil.copyfile(os.path.join(times_optimizer.best_output,
                                         os.path.split(model_path[1])[1]),
                            self.wait_model_path)
        else:
            # Copy best model to destination folder
            source = os.path.join(times_optimizer.best_output,
                              self.parms['file'].split('.')[0]+'.h5')
            shutil.copyfile(source, self.model_path)
        # Save metadata
        sup.create_json(model_metadata, metadata_file)

        
# =============================================================================
# Train model
# =============================================================================
    def extract_distribution(self, X):
        f = Fitter(X,
            distributions=['norm', 'expon', 'uniform', 'lognorm', 'loguniform'])
        f.fit()
        f.summary()
        return list(f.get_best(method = 'sumsquare_error').keys())[0]

    def extract_day_moment(self, start_timestamp):
        if start_timestamp.hour >= 0 and start_timestamp.hour < 12:
            return 'morning'
        elif start_timestamp.hour >= 12 and start_timestamp.hour < 17:
            return 'afternoon'
        elif start_timestamp.hour >= 17 and start_timestamp.hour < 24:
            return 'night'
    
    
    def extract_description_activities(self, log):
        log['order'] = log.sort_values(by='start_timestamp', ascending=True).groupby('caseid').cumcount() + 1
        activity_desc = []
        for activity in log['task'].drop_duplicates():
            log_activity = log[log['task'] == activity]
            day_moment = list(set([self.extract_day_moment(x) for x in log_activity['start_timestamp']]))
            rol = list(set([x for x in log_activity['user']]))
            trace_position = np.mean(list(set([x for x in log_activity['order']])))
            distribution = self.extract_distribution([x for x in log_activity['processing_time']])
            mean_proc_time = np.mean(log_activity['processing_time'])
            std_proc_time = np.std(log_activity['processing_time'])
            activity_desc.append([activity, day_moment, rol, trace_position, distribution, mean_proc_time, std_proc_time])

        df_activity_desc = pd.DataFrame(data=activity_desc, columns = ['task_name', 'day_moment', 'rol', 'trace_position', 'dstribution', 'mean_processing_time', 'std_processing_time'])
        df_activity_desc.to_csv('output_files/Activity_description.csv', sep='|')

    def _discover_model(self, **kwargs):
        # indexes creation
        self.ac_index, self.index_ac = self._indexing(self.log.data, 'task')
        self.usr_index, self.index_usr = self._indexing(self.log.data, 'user')
        # replay
        self._replay_process()
        
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            self._add_intercases()
        self._split_timeline(0.8, self.one_timestamp)
        self.log_train = self._add_calculated_times(self.log_train)
        self.log_valdn = self._add_calculated_times(self.log_valdn)
        # Add index to the event log
        ac_idx = lambda x: self.ac_index[x['task']]
        self.log_train['ac_index'] = self.log_train.apply(ac_idx, axis=1)
        self.log_valdn['ac_index'] = self.log_valdn.apply(ac_idx, axis=1)
        if self.parms['model_type'] in ['inter_nt', 'dual_inter']:
            ac_idx = lambda x: self.ac_index[x['n_task']]
            self.log_train['n_ac_index'] = self.log_train.apply(ac_idx, axis=1)
            self.log_valdn['n_ac_index'] = self.log_valdn.apply(ac_idx, axis=1)
        # Load embedding matrixes
        self.train_val_log = pd.concat([pd.DataFrame(self.log_train), pd.DataFrame(self.log_valdn)])
        self.ac_index_train_val, self.index_ac_train_val = self._indexing(self.train_val_log, 'task')
        self.usr_index_train_val, self.index_usr_train_val = self._indexing(self.train_val_log, 'user')

        emb_trainer = emb.Embedder(self.parms,
                                    self.log,
                                    self.ac_index,
                                    self.index_ac,
                                    self.usr_index,
                                    self.index_usr)
                                    
        self.ac_weights = emb_trainer.Embedd(self.parms['emb_method'])
        self.extract_description_activities(self.log.copy())
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
        return times_optimizer

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
        self.log = replayer.process_stats.rename(columns={'resource':'user'})
        self.log['user'] = self.log['user'].fillna('sys')
        self.log = self.log.to_dict('records')
    

    @staticmethod
    def _indexing(log, feat):
        log = pd.DataFrame(log)
        # Activities index creation
        if feat=='task':
            log = log[~log[feat].isin(['Start', 'End'])]
        else:
            log[feat] = log[feat].fillna('sys')
        subsec_set = log[feat].unique().tolist()
        subsec_set = [x for x in subsec_set if not x in ['Start', 'End']]
        index = dict()
        for i, _ in enumerate(subsec_set):
            index[subsec_set[i]] = i + 1
        index['Start'] = 0
        index['End'] = len(index)
        index_inv = {v: k for k, v in index.items()}
        return index, index_inv

    
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

    def _add_intercases(self):
        """Appends the indexes and relative time to the dataframe.
        parms:
            log: dataframe.
        Returns:
            Dataframe: The dataframe with the calculated features added.
        """
        log = pd.DataFrame(self.log)
        res_analyzer = rl.ResourcePoolAnalyser(
            log,
            sim_threshold=self.parms['rp_similarity'])
        resource_table = pd.DataFrame.from_records(res_analyzer.resource_table)
        resource_table.rename(columns={'resource': 'user'}, inplace=True)
        self.roles = {role: group.user.to_list() for role, group in resource_table.groupby('role')}
        log = log.merge(resource_table, on='user', how='left')
        inter_mannager = it.IntercaseMannager(log, 
                                              self.parms['all_r_pool'],
                                              self.parms['model_type'])
        log, mean_states = inter_mannager.fit_transform()
        self.mean_states = mean_states
        self.log = log
        roles_table = (self.log[['caseid', 'role', 'task']]
                .groupby(['task', 'role']).count()
                .sort_values(by=['caseid'])
                .groupby(level=0)
                .tail(1)
                .reset_index())
        self.roles_table = roles_table[['role', 'task']]


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
                events[i]['st_daytime'] = time
                events[i]['st_weekday'] = events[i]['start_timestamp'].weekday()
                if self.parms['model_type'] == 'dual_inter':
                    time = events[i]['end_timestamp'].time()
                    time = time.second + time.minute*60 + time.hour*3600
                    events[i]['end_daytime'] = time
                    events[i]['end_weekday'] = events[i]['end_timestamp'].weekday()
        return pd.DataFrame.from_dict(log)

    def _transform_features(self):
        # scale continue features
        cols = ['processing_time', 'waiting_time']
        self.scaler = MaxAbsScaler()
        self.scaler.fit(self.log_train[cols])
        self.log_train[cols] = self.scaler.transform(self.log_train[cols])
        self.log_valdn[cols] = self.scaler.transform(self.log_valdn[cols])
        # scale intercase
        # para que se ajuste a los dos sets requeridos para el modelo dual
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            inter_feat = ['st_wip', 'st_tsk_wip']
            self.inter_scaler = MaxAbsScaler()
            self.inter_scaler.fit(self.log_train[inter_feat])
            self.log_train[inter_feat] = (
                self.inter_scaler.transform(self.log_train[inter_feat]))
            self.log_valdn[inter_feat] = (
                self.inter_scaler.transform(self.log_valdn[inter_feat]))
            cols.extend(inter_feat)
            if self.parms['model_type'] in ['dual_inter']:
                inter_feat = ['end_wip']
                self.end_inter_scaler = MaxAbsScaler()
                self.end_inter_scaler.fit(self.log_train[inter_feat])
                self.log_train[inter_feat] = (
                    self.end_inter_scaler.transform(self.log_train[inter_feat]))
                self.log_valdn[inter_feat] = (
                    self.end_inter_scaler.transform(self.log_valdn[inter_feat]))
                cols.extend(inter_feat)
        # scale daytime
        self.log_train['st_daytime'] = np.divide(self.log_train['st_daytime'], 86400)
        self.log_valdn['st_daytime'] = np.divide(self.log_valdn['st_daytime'], 86400)
        cols.extend(['caseid', 'ac_index', 'st_daytime', 'st_weekday'])
        if self.parms['model_type'] in ['dual_inter']:
            self.log_train['end_daytime'] = np.divide(self.log_train['end_daytime'], 86400)
            self.log_valdn['end_daytime'] = np.divide(self.log_valdn['end_daytime'], 86400)
            cols.extend(['end_weekday'])
        if self.parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']:
            suffixes = (['_st_oc', '_end_oc'] if (
                self.parms['model_type']  in ['dual_inter']) else ['_st_oc'])
            if self.parms['all_r_pool']:
                for suffix in suffixes:
                    cols.extend([c_n for c_n in self.log_train.columns 
                                 if suffix in c_n])
            else:
                cols.extend(['rp'+x for x in suffixes])
        # Add next activity
        if self.parms['model_type'] in ['inter_nt', 'dual_inter']:
            cols.extend(['n_ac_index'])
        # filter features
        self.log_train = self.log_train[cols]
        self.log_valdn = self.log_valdn[cols]
        # fill nan values
        self.log_train = self.log_train.fillna(0)
        self.log_valdn = self.log_valdn.fillna(0)
