# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:48:27 2021

@author: Manuel Camargo
"""
import random
import numpy as np

from collections import deque
import uuid

from keras.utils import np_utils as ku
from operator import itemgetter
from queue import PriorityQueue
import queue


class Queue():
    """
    """
    def __init__(self):
        """constructor"""
        self._queue = PriorityQueue()

    def add(self, element):
        self._queue.put((element['timestamp'], id(element), element))

    def get_remove_first(self):
        try:
            return self._queue.get(block=False)[2]
        except queue.Empty:
            return list()

    def get_all(self):
        return self._queue


# class Role():

#     def __init__(self, name, size, index=0, check_avail=True):
#         self._num_resources = size
#         self._resource_pool = self._initialize_resources(size)
#         self._name = name
#         self._index = index
#         self._check_avail = check_avail
#         self._execution = list() if check_avail else 0
        
#     def assign_resource(self, release_time):
#         if self._check_avail:
#             try:
#                 res = random.choice(self._resource_pool)
#                 self._resource_pool.remove(res)
#                 res['release_time'] = release_time
#                 self._execution.append(res)
#                 self._execution = sorted(self._execution, key=itemgetter('release_time'))
#                 return res['resource']
#             except IndexError:
#                 return None
#         else:
#             res = random.choice(self._resource_pool)
#             self._execution += 1
#             return res['resource']
        
#     def release_resource(self):
#         if self._check_avail:
#             try:
#                 res = self._execution.pop(0)
#                 res['release_time'] = None
#                 self._resource_pool.append(res)
#                 return res['resource']
#             except IndexError:
#                 return None
#         else:
#             self._execution -= 1

#     def get_occupancy(self):
#         if self._check_avail:
#             return len(self._execution)/self._num_resources
#         else:
#             occ = self._execution/self._num_resources
#             return occ if occ < 1 else 1
    
#     def get_availability(self):
#         free_r = len(self._resource_pool)
#         return free_r if free_r >=0 else 0
    
#     def get_name(self):
#         return self._name
    
#     def get_resource_pool(self):
#         return self._resource_pool
    
#     def get_execution(self):
#         return self._execution
    
#     def get_next_release(self):
#         try:
#             next_release = self._execution[0]
#             return next_release['release_time']
#         except IndexError:
#             return None
        
#     @staticmethod
#     def _initialize_resources(size):
#         resource_pool = deque()
#         for num in range(0, size):
#             resource_pool.append({'resource': 'res_'+str(uuid.uuid4()),
#                                   'release_time': None})
#         return resource_pool
class Role():

    def __init__(self, name, size, index=0, check_avail=True):
        self._num_resources = size
        self._resource_pool = self._initialize_resources(size)
        self._name = name
        self._index = index
        self._check_avail = check_avail
        self._execution = 0
        
    def assign_resource(self, release_time):
        if self._check_avail:
            try:
                avail_resources = [k for k, v in self._resource_pool.items() 
                                   if v.get('release_time') is None]
                res_id = random.choice(avail_resources)
                self._resource_pool.get(res_id)['release_time'] = release_time
                return res_id
            except IndexError:
                return None
        else:
            res_id = random.choice(list(self._resource_pool.keys()))
            self._execution += 1
            return res_id
        
    def release_resource(self, res_id):
        if self._check_avail:
            try:
                self._resource_pool.get(res_id)['release_time'] = None
            except KeyError:
                return 'Unexistent resource'
        else:
            self._execution -= 1

    def get_occupancy(self):
        if self._check_avail:
            occupied = {k: v for k, v in self._resource_pool.items() 
                        if v.get('release_time') is not None}
            return len(occupied)/self._num_resources
        else:
            occ = self._execution/self._num_resources
            return occ if occ < 1 else 1
    
    def get_availability(self):
        free_r = [k for k, v in self._resource_pool.items() 
                  if v.get('release_time') is None]
        return free_r if free_r >=0 else 0
    
    def get_name(self):
        return self._name
    
    def get_resource_pool(self):
        return self._resource_pool
    
    def get_execution(self):
        return self._execution
    
    def get_next_release(self):
        try:
            next_release = min([
                v.get('release_time') for k, v in self._resource_pool.items() 
                if v.get('release_time') is not None])
            return next_release
        except ValueError:
            return None
        
    @staticmethod
    def _initialize_resources(size):
        resource_pool = dict()
        for num in range(0, size):
            resource_pool['res_'+str(uuid.uuid4())] = {'release_time': None}
        return resource_pool


class ActivityCounter():

    def __init__(self, name, index=0, initial=0):
        self._name = name
        self._index = index
        self._active_instances = initial

    def add_act(self):
        self._active_instances += 1

    def remove_act(self):
        self._active_instances -= 1

    def get_active_instances(self):
        return self._active_instances

    def get_name(self):
        return self._name


class ProcessInstance():

    def __init__(self, cid, n_size, n_features, n_act=False, dual=False):
        self._id = cid
        if dual:
            self.init_dual_ngram(n_size, *n_features)
        else:
            self.init_ngram(n_size, n_features, n_act)
        self.proc_t, self.wait_t = 0.0, 0.0
        
    def init_ngram(self, n_size, n_features, n_act):
        self._act_ngram = [0 for i in range(n_size)]
        self._feat_ngram = np.zeros((1, n_size, n_features))
        if n_act:
            self._n_act_ngram = [0 for i in range(n_size)]
        
    def init_dual_ngram(self, n_size, n_feat_proc, n_feat_wait):
        self._act_ngram = [0 for i in range(n_size)]
        self._n_act_ngram = [0 for i in range(n_size)]
        self._proc_feat_ngram = np.zeros((1, n_size, n_feat_proc))
        self._wait_feat_ngram = np.zeros((1, n_size, n_feat_wait))
        
    def get_ngram(self, n_act=False):
        if n_act:
            return self._act_ngram, self._n_act_ngram, self._feat_ngram
        else:
            return self._act_ngram, self._feat_ngram
    
    def get_proc_ngram(self):
        return self._act_ngram, self._proc_feat_ngram
    
    def get_wait_ngram(self):
        return self._n_act_ngram, self._wait_feat_ngram

    def update_ngram(self, ac, ts, wip, rp_oc, n_act=None):
        '''
        feature order: weekday, proc_t, wait_t, pr_instances
        tsk_start_inst, daytime, rp_start_oc
        '''
        daytime = ts.time()
        daytime = daytime.second + daytime.minute*60 + daytime.hour*3600
        daytime = daytime / 86400
        day_dummies = ku.to_categorical(ts.weekday(), num_classes=7)
        record = (list(day_dummies) +
                  [self.proc_t, self.wait_t] +
                  list(wip) +
                  [daytime] +
                  rp_oc)
        self._feat_ngram = np.append(self._feat_ngram,
                                     np.array([[record]], dtype=object), axis=1)
        self._feat_ngram = np.delete(self._feat_ngram, 0, 1)
        self._act_ngram.append(ac)
        self._act_ngram.pop(0)
        if n_act is not None:
            self._n_act_ngram.append(n_act)
            self._n_act_ngram.pop(0)
        
    def update_proc_ngram(self, ac, ts, wip, rp_oc):
        '''
        feature order: weekday, proc_t, wait_t, pr_instances
        tsk_start_inst, daytime, rp_start_oc
        '''
        daytime = ts.time()
        daytime = daytime.second + daytime.minute*60 + daytime.hour*3600
        daytime = daytime / 86400
        day_dummies = ku.to_categorical(ts.weekday(), num_classes=7)
        record = ([self.proc_t] +
                  list(wip) +
                  [daytime] +
                  rp_oc +
                  list(day_dummies))
        self._proc_feat_ngram = np.append(self._proc_feat_ngram,
                                          np.array([[record]], dtype=object),
                                          axis=1)
        self._proc_feat_ngram = np.delete(self._proc_feat_ngram,
                                          0, 1)
        self._act_ngram.append(ac)
        self._act_ngram.pop(0)

    def update_wait_ngram(self, nac, ts, wip, rp_oc):
        '''
        feature order: weekday, proc_t, wait_t, pr_instances
        tsk_start_inst, daytime, rp_start_oc
        '''
        # daytime = ts.time()
        # daytime = daytime.second + daytime.minute*60 + daytime.hour*3600
        # daytime = daytime / 86400
        day_dummies = ku.to_categorical(ts.weekday(), num_classes=7)
        record = ([self.wait_t] +
                  list(wip) +
                  rp_oc +
                  list(day_dummies))
        self._wait_feat_ngram = np.append(self._wait_feat_ngram,
                                          np.array([[record]], dtype=object),
                                          axis=1)
        self._wait_feat_ngram = np.delete(self._wait_feat_ngram, 0, 1)
        self._n_act_ngram.append(nac)
        self._n_act_ngram.pop(0)

    def update_proc_wait(self, proc_t, wait_t):
        self.proc_t = proc_t
        self.wait_t = wait_t

    def update_proc(self, proc_t):
        self.proc_t = proc_t

    def update_wait(self, wait_t):
        self.wait_t = wait_t

