# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:22:42 2021

@author: Manuel Camargo
"""
import os
import json
import numpy as np
import pandas as pd

import warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
from core_modules.times_allocator import entities as en
from nltk import ngrams

from datetime import timedelta
import uuid
from tqdm import tqdm


from pickle import load
from enum import Enum

class InstanceState(Enum):
    WAITING = 1
    INEXECUTION = 2
    COMPLETE = 3


class IntercasesPredictor():
    """
    """
    def __init__(self, model_path, parms):
        """constructor"""
        tf.compat.v1.reset_default_graph()
        tf.keras.backend.clear_session()
        # load model
        self.g, self.session, self.model = self._load_models(model_path)
        # self.model = load_model(model_path)
        self.parms = parms
        self.n_features = self.model.get_layer('features').output_shape[0][2]
        # next activity
        self.n_act = self.parms['model_type'] == 'inter_nt'

    def _load_models(self, path):
        graph = tf.Graph()
        with graph.as_default():
            session = tf.compat.v1.Session()
            with session.as_default():            
                model = load_model(path)
        return graph, session, model

    def predict(self, sequences, iarr):
        if self.n_act:
            tmodel = '_inapr' if self.parms['all_r_pool'] else '_inspr'
        else:
            tmodel = '_iapr' if self.parms['all_r_pool'] else '_ispr'
        metadata_file = os.path.join(
            self.parms['times_gen_path'],
            self.parms['file'].split('.')[0]+tmodel+'_meta.json')
        # Loading of parameters from existing model
        if os.path.exists(metadata_file):
            with open(metadata_file) as file:
                data = json.load(file)
                self.ac_index = data['ac_index']
                self.index_ac = {v: k for k, v in self.ac_index.items()}
                n_size = data['n_size']
                rl_task = pd.DataFrame(data['roles_table'])
                rl_table = pd.DataFrame([{'role_name': item[0],
                                          'size': len(item[1]), 
                                          'role_index': num} 
                                  for num, item in enumerate(data['roles'].items())])
                pr_act_initial = int(
                    round(data['inter_mean_states']['wip']))
                init_states = data['inter_mean_states']['tasks']
        # loading scalers
        s_path = os.path.join(self.parms['times_gen_path'], 
                              self.parms['file'].split('.')[0]+tmodel+'_scaler.pkl')
        i_s_path = os.path.join(self.parms['times_gen_path'],
                                self.parms['file'].split('.')[0]+tmodel+'_inter_scaler.pkl')
        self.scaler = load(open(s_path, 'rb'))
        self.inter_scaler = load(open(i_s_path, 'rb'))
        # Initialize values
        iarr = {x['caseid']: x['timestamp'] for x in iarr.to_dict('records')}
        sequences = sequences[~sequences.task.isin(['Start', 'End'])]
        self.sequences, num_elements = self._encode_secuences(
            sequences, self.ac_index, rl_task, rl_table, self.n_act)
        # Init resource pools
        self.rl_dict = self._initialize_roles(
            rl_table, check_avail=self.parms['reschedule'])
        # Init activities
        self.ac_dict = self._initialize_activities(self.ac_index, init_states)
        self.queue = self._initialize_queue(iarr)
        self.execution_state = self._initialize_exec_state(self.sequences)
        # [print(x) for x in self.execution_state]
        return self._generate(pr_act_initial, n_size, num_elements)

        
    
    def _generate(self, pr_act_initial, n_size, num_elements):
        event_log = list()
        def create_record(cid, ac_rl, res, ts):
            ac_rl = ac_rl[0] if self.n_act else ac_rl
            return {'caseid': cid,
                    'task': self.index_ac[ac_rl[0]],
                    'resource': res,
                    'role': self.rl_dict[ac_rl[1]].get_name(),
                    'end_timestamp': ts}
        
        open_events = dict()
        active_instances = dict()
        pr_wip = pr_act_initial
        
        pbar = tqdm(total=num_elements, desc='generating traces:')
        while not self.queue.get_all().empty():
            element = self.queue.get_remove_first()
            cid = element['caseid']
            if element['action'] == 'create_instance':
                transition = self.execution_state[cid]['transitions'].pop(0)
                self.execution_state[cid]['state'] = InstanceState.INEXECUTION
                self.queue.add({
                    'timestamp': element['timestamp'],
                    'action': 'create_activity',
                    'caseid': cid,
                    'transition': transition})
                # add active instances
                pr_wip += 1
                # initialize instance
                active_instances[cid] = en.ProcessInstance(
                    cid, n_size, self.n_features, n_act=self.n_act)
                # print(cid)
            elif element['action'] == 'create_activity':
                # print(cid)
                transition = element['transition']
                ac = transition[0][0] if self.n_act else transition[0]
                rl = transition[0][1] if self.n_act else transition[1]
                # encode record
                ac_wip = self.ac_dict[ac].get_active_instances()
                if self.parms['all_r_pool']:
                    rp_oc = [self.rl_dict[x].get_occupancy() 
                             for x in range(0, len(self.rl_dict))]
                else:
                    rp_oc = [self.rl_dict[rl].get_occupancy()]
                wip = self.inter_scaler.transform(np.array([pr_wip, ac_wip])
                    .reshape(-1, 2))[0]
                
                n_ac = transition[1][0] if self.n_act else None
                active_instances[cid].update_ngram(ac,
                                                   element['timestamp'],
                                                   wip, rp_oc,
                                                   n_act=n_ac)
                # predict
                if self.n_act:
                    act_ngram, n_act_ngram, feat_ngram = active_instances[cid].get_ngram(self.n_act)
                    model_input = {'ac_input': np.array([act_ngram]),
                                   'n_ac_input': np.array([n_act_ngram]),
                                   'features': feat_ngram}                
                else:
                    act_ngram, feat_ngram = active_instances[cid].get_ngram()
                    model_input = {'ac_input': np.array([act_ngram]),
                                   'features': feat_ngram}
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    with self.g.as_default():
                        with self.session.as_default():
                            preds = self.model.predict(model_input)
                preds[preds < 0] = 0.000001
                proc_t, wait_t = preds[0]
                active_instances[cid].update_proc_wait(proc_t, wait_t)
                ipred = self.scaler.inverse_transform(preds)
                iproc_t, iwait_t = ipred[0]
                # resource assignment
                release_time = element['timestamp'] + timedelta(
                    seconds=int(round(iproc_t)))
                res = self.rl_dict[rl].assign_resource(release_time)
                # verify availability and reshedule if needed
                if res is not None:
                    ev_id = 'event_'+str(uuid.uuid4())
                    open_events[ev_id] = dict()
                    open_events[ev_id]['res_id'] = res
                    open_events[ev_id]['pr_instances'] = pr_wip
                    open_events[ev_id]['tsk_start_inst'] = ac_wip
                    if not self.parms['all_r_pool']:
                        open_events[ev_id]['rp_start_oc'] = rp_oc[0]
                    open_events[ev_id]['start_timestamp'] = element['timestamp']
                    # actualizar instancias de actividad activas
                    self.ac_dict[ac].add_act()
                    # save time
                    element['timestamp'] = release_time
                    element['action'] = 'complete_activity'
                    element['n_act'] = int(round(iwait_t))
                    element['ev_id'] = ev_id
                else:
                    release_time = self.rl_dict[rl].get_next_release()
                    element['timestamp'] = release_time + timedelta(microseconds=1)
                self.queue.add(element)
            elif element['action'] == 'complete_activity':
                transition = element['transition']
                ac = transition[0][0] if self.n_act else transition[0]
                rl = transition[0][1] if self.n_act else transition[1]
                # get event info
                event = open_events[element['ev_id']]
                # desasignar recurso
                self.rl_dict[rl].release_resource(event['res_id'])
                # actualizar instancias de actividad activas
                self.ac_dict[ac].remove_act()
                # save time
                event = {**create_record(cid, transition, event['res_id'], 
                                         element['timestamp']), **event}
                event.pop('res_id', None)
                event_log.append(event)
                open_events.pop(element['ev_id'], None)
                # si es la ultima transicion agende complete instance
                try:
                    element['transition'] = (
                        self.execution_state[cid]['transitions'].pop(0))
                    element['timestamp'] += timedelta(seconds=element['n_act'])
                    element['action'] = 'create_activity'
                except IndexError:
                    element['action'] = 'complete_instance'
                pbar.update(1)
                self.queue.add(element)
            elif element['action'] == 'complete_instance':
                self.execution_state[cid]['state']=InstanceState.COMPLETE
                # substract instancias activas
                pr_wip -= 1
                active_instances.pop(cid, None)
            else:
                raise ValueError('Unexistent action')
        pbar.close()
        return event_log

        
    @staticmethod
    def _initialize_activities(ac_index, init_states):        
        activities_dict = dict()
        for key, value in ac_index.items():
            if key not in ['Start', 'End']:
                activities_dict[value] = en.ActivityCounter(
                    key, index=value, initial=int(round(init_states.get(key, 0))))
        return activities_dict
    
    @staticmethod
    def _initialize_roles(roles, check_avail):        
        rl_dict = dict()
        for role in roles.to_dict('records'):
            rl_dict[role['role_index']] = en.Role(role['role_name'],
                                                  role['size'],
                                                  index=role['role_index'],
                                                  check_avail=check_avail)
        return rl_dict
    
    @staticmethod
    def _initialize_queue(iarr):       
        queue = en.Queue()
        for k, v in iarr.items():
            queue.add({'timestamp': v,
                       'action': 'create_instance',
                       'caseid': k})
        return queue

    @staticmethod
    def _initialize_exec_state(sequences):
        execution_state = dict()
        for k, transitions in sequences.items():
            execution_state[k] = {'state': InstanceState.WAITING, 
                                  'transitions': transitions}
        return execution_state
    
    @staticmethod
    def _encode_secuences(sequences, ac_idx, rl_task, rl_table, n_act):
        seq = sequences.copy()
        # Determine biggest resource pool as default 
        def_role = rl_table[rl_table['size']==rl_table['size'].max()].iloc[0]['role_name']
        # Assign roles to activities 
        seq['ac_index'] = seq.apply(lambda x: ac_idx[x.task], axis=1)
        seq = seq.merge(rl_task, how='left', on='task')
        seq.fillna(value={'role': def_role}, inplace=True)
        seq = seq.merge(rl_table, how='left', left_on='role', right_on='role_name')
        ac_rl = lambda x: (x.ac_index, x.role_index)
        seq['ac_rl'] = seq.apply(ac_rl, axis=1)
        num_elements = 0
        encoded_seq = dict()
        for key, group in seq.sort_values('pos_trace').groupby('caseid'):
            if n_act:
                encoded_seq[key] = list(
                    ngrams(group.ac_rl.to_list(), 2, pad_right=True,
                           right_pad_symbol=(ac_idx['End'], 0)))
            else:
                encoded_seq[key] = group.ac_rl.to_list()
            num_elements += len(encoded_seq[key])
        return encoded_seq, num_elements
