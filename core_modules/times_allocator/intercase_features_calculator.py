# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import numpy as np
import pandas as pd

import itertools as it
from datetime import timedelta


class RoleCounter():

    def __init__(self, name, size):
        self._num_resources = size
        self._name = name
        self._asigned = 0

    def assign_resource(self):
        self._asigned += 1

    def release_resource(self):
        self._asigned -= 1
        
    def get_occupancy(self):
        return self._asigned/self._num_resources
    
    def get_availability(self):
        free_r = self._num_resources-self._asigned
        return free_r if free_r >=0 else 0
    
    def get_name(self):
        return self._name

class ActivityCounter():

    def __init__(self, name):
        self._name = name
        self._active_instances = 0

    def add_act(self):
        self._active_instances += 1

    def remove_act(self):
        self._active_instances -= 1
        
    def get_active_instances(self):
        return self._active_instances
    
    def get_name(self):
        return self._name


class IntercaseMannager():
    
    def __init__(self, log, all_r_pool, model_type):
        self._log = log
        self.all_r_pool = all_r_pool
        self.model_type = model_type
    
    def fit_transform(self):
        self._append_csv_start_end()
        event_log = self._create_expanded_log()
        rl_ocup, wip, tsk_wip = self._calculate_feature(event_log)
        log = self._extend_original_log(rl_ocup, wip, tsk_wip)
        log = (log[(log.task != 'Start') & (log.task != 'End')]
               .reset_index(drop=True))
        mean_states = {
            'tasks': { row.task: row.task_instances 
                      for k, row in tsk_wip.groupby('task')
                      .task_instances.mean()
                      .reset_index().iterrows()},
            'wip': np.mean(wip.wip)}
        return log, mean_states
    
    def _append_csv_start_end(self):
        end_start_times = dict()
        for case, group in self._log.groupby('caseid'):
            end_start_times[(case, 'Start')] = (
                group.start_timestamp.min()-timedelta(microseconds=1))
            end_start_times[(case, 'End')] = (
                group.end_timestamp.max()+timedelta(microseconds=1))
        new_data = list()
        data = sorted(self._log.to_dict('records'), key=lambda x: x['caseid'])
        for key, group in it.groupby(data, key=lambda x: x['caseid']):
            trace = list(group)
            for new_event in ['Start', 'End']:
                idx = 0 if new_event == 'Start' else -1
                temp_event = dict()
                temp_event['caseid'] = trace[idx]['caseid']
                temp_event['task'] = new_event
                temp_event['user'] = new_event
                temp_event['end_timestamp'] = end_start_times[(key, new_event)]
                temp_event['start_timestamp'] = end_start_times[(key, new_event)]
                if new_event == 'Start':
                    trace.insert(0, temp_event)
                else:
                    trace.append(temp_event)
            new_data.extend(trace)
        self._log = pd.DataFrame(new_data)

    def _create_expanded_log(self):
        starts = self._log.copy()
        starts = starts.drop(columns=['end_timestamp'])
        starts.rename(columns={'start_timestamp': 'timestamp'}, inplace=True) 
        starts['transition'] = 'start'
        complete = self._log.copy()
        complete = complete.drop(columns=['start_timestamp'])
        complete.rename(columns={'end_timestamp': 'timestamp'}, inplace=True) 
        complete['transition'] = 'complete'
        event_log = pd.concat([starts, complete], axis=0).reset_index(drop=True)
        event_log.sort_values(['timestamp', 'transition'], ascending=[True, False], inplace=True)
        event_log.reset_index(drop=True, inplace=True)
        return event_log
        
    def _calculate_feature(self, event_log):
        roles_dict = self._initialize_roles(self._log)
        activities_dict = self._initialize_activities(self._log)     

        process_instances = list()
        respool_ocupancies = list()
        tasks_active_instances = list() 
        process_active_instances = 0
        for ts, group in event_log.groupby('timestamp'):
            for idx, event in group.iterrows():
                if event.task in ['Start', 'End']:
                    if event.task =='Start' and event.transition =='start':
                        process_active_instances +=1
                    elif event.task =='End' and event.transition =='start':
                        process_active_instances -=1
                else:
                    if (not pd.isnull(event.role)) and (event.transition =='start'):
                        roles_dict[event.role].assign_resource()
                        activities_dict[event.task].add_act()
                    elif (not pd.isnull(event.role)) and (event.transition =='complete'):
                        roles_dict[event.role].release_resource()
                        activities_dict[event.task].remove_act()
            process_instances.append({'timestamp': ts, 
                              'wip': process_active_instances})
            respool_ocupancies.extend([
                {'timestamp': ts,
                 'role': role.get_name(),
                 'respool_oc': role.get_occupancy()} 
                for role in roles_dict.values()])
            tasks_active_instances.extend([
                {'timestamp': ts,
                 'task': task.get_name(),
                 'task_instances': task.get_active_instances()} 
                for task in activities_dict.values()])
        respool_ocupancies = pd.DataFrame(respool_ocupancies)
        process_active_instances = pd.DataFrame(process_instances)
        tasks_active_instances = pd.DataFrame(tasks_active_instances)
        return respool_ocupancies, process_active_instances, tasks_active_instances

    @staticmethod
    def _initialize_activities(log):        
        activities_dict = dict()
        for key in log.task.unique():
            activities_dict[key] = ActivityCounter(key)
        return activities_dict
    
    @staticmethod
    def _initialize_roles(log):        
        roles_dict = dict()
        for key, group in log[['role', 'user']].drop_duplicates().groupby('role'):
            roles_dict[key] = RoleCounter(key, len(group))
        return roles_dict
    
    def _extend_original_log(self, rl_ocup, wip, tsk_wip):
        # initial wip
        log2 = self._log.merge(wip,
                               how='left',
                               left_on='start_timestamp', 
                               right_on='timestamp')
        log2.rename(columns={'wip': 'st_wip'}, inplace=True)
        log2.drop(columns=['timestamp'], inplace=True)
        log2 = log2.merge(tsk_wip, 
                         how='left', 
                         left_on=['start_timestamp', 'task'], 
                         right_on=['timestamp', 'task'])
        log2.drop(columns=['timestamp'], inplace=True)
        log2.rename(columns={'task_instances': 'st_tsk_wip'}, inplace=True)
        # Next activity
        log2['n_task'] = (log2['task']
                          .groupby(log2['caseid'])
                          .transform(lambda x: x.shift(-1)))
        if self.all_r_pool:
            # initial rpool_occ
            rl_ocup = pd.pivot_table(rl_ocup, values='respool_oc', 
                                     columns='role', index=['timestamp'],
                                     fill_value=0).reset_index()
            r_pools = {name: name.lower().replace(" ", "_")+'_st_oc' 
                       for name in list(rl_ocup.columns) if name != 'timestamp'}
            log2 = log2.merge(rl_ocup, 
                             how='left', 
                             left_on=['start_timestamp'], 
                             right_on=['timestamp'])
            log2.drop(columns=['timestamp'], inplace=True)
            log2.rename(columns=r_pools, inplace=True)
            if self.model_type == 'dual_inter':
                # final wip
                log2 = log2.merge(wip, 
                                  how='left', 
                                  left_on='end_timestamp', 
                                  right_on='timestamp')
                log2.rename(columns={'wip': 'end_wip'}, inplace=True)
                log2.drop(columns=['timestamp'], inplace=True)
                # final rpool_occ
                r_pools = {name: name.lower().replace(" ", "_")+'_end_oc' 
                           for name in list(rl_ocup.columns) if name != 'timestamp'}
                log2 = log2.merge(rl_ocup, 
                                 how='left', 
                                 left_on=['end_timestamp'], 
                                 right_on=['timestamp'])
                log2.drop(columns=['timestamp'], inplace=True)
                log2.rename(columns=r_pools, inplace=True)
        else:
            log2 = log2.merge(rl_ocup, 
                             how='left', 
                             left_on=['start_timestamp', 'role'], 
                             right_on=['timestamp', 'role'])
            log2.drop(columns=['timestamp'], inplace=True)
            log2.rename(columns={'respool_oc': 'rp_st_oc'}, inplace=True)
            if self.model_type == 'dual_inter':
                # final wip
                log2 = log2.merge(wip, 
                                  how='left', 
                                  left_on='end_timestamp', 
                                  right_on='timestamp')
                log2.rename(columns={'wip': 'end_wip'}, inplace=True)
                log2.drop(columns=['timestamp'], inplace=True)
                # Next role 
                log2['n_role'] = (log2['role']
                                  .groupby(log2['caseid'])
                                  .transform(lambda x: x.shift(-1)))
                log2 = log2.merge(rl_ocup, 
                                 how='left', 
                                 left_on=['end_timestamp', 'n_role'], 
                                 right_on=['timestamp', 'role'])
                log2.drop(columns=['timestamp', 'role_y'], inplace=True)
                log2.rename(columns={'respool_oc': 'rp_end_oc',
                                     'role_x': 'role'}, inplace=True)
                log2.fillna(value={'rp_end_oc': 0}, inplace=True)
        return log2
