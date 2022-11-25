# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:40:36 2021

@author: Manuel Camargo
"""

# -*- coding: utf-8 -*-
from extraction import role_discovery as rd
# from support_modules import role_discovery as rd
import gensim
import os
import pandas as pd
import numpy as np
import utils.support as sup
import math
import itertools
from datetime import datetime

"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""


class EmbeddingWord2vec():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        # """Main method of the embedding training module.
        # Args:
        #     parameters (dict): parameters for training the embeddeding network.
        #     timeformat (str): event-log date-time format.
        #     no_loops (boolean): remove loops fom the event-log (optional).
        # """
        # Define the number of dimensions as the 4th root of the # of categories
        self.log = log.copy()
        self.ac_index = ac_index
        self.index_ac = index_ac
        self.usr_index = usr_index
        self.index_usr = index_usr
        self.file_name = params['file']
        self.embedded_path = params['embedded_path']
        self.concat_method = params['concat_method']
        self.include_times = params['include_times']

    def load_embbedings(self):
        # Embedded dimensions
        self.ac_weights = list()
        self.ac_weights2 = list()
        self.ac_weights3 = list()

        print(self.ac_index)
        print('Number of instances in embeddings input: {}'.format(len(self.log['caseid'].drop_duplicates())))

        # Load embedded matrix
        ac_emb = 'ac_W2V_' + '{}_'.format(self.concat_method) + self.file_name.split('.')[0]  +'.emb'
        if os.path.exists(os.path.join(self.embedded_path, ac_emb)):
            return self._read_embedded(self.index_ac, ac_emb)
        else:
            self.log['duration'] = self.log.apply(lambda x: (x['end_timestamp'] - x['start_timestamp']).total_seconds(), axis= 1)

            n_bins = int((np.max(self.log['duration']) - np.min(self.log['duration']))/(np.mean(self.log['duration'])))
            print('The number of intervals are: {}'.format(n_bins))
            self.log['duration_cat'] = pd.qcut(self.log['duration'], n_bins, labels=False, duplicates = 'drop')

            dim_number = math.ceil(
                len(list(itertools.product(*[list(self.ac_index.items()),
                                             list(self.usr_index.items())]))) ** 0.25)
            
            self.extract_log_info()

            if self.concat_method == 'full_sentence':
                self.extract_full_sentence()
                EmbeddingWord2vec.learn(self, self.input_data, dim_number)
            elif self.concat_method == 'single_sentence':
                self.extract_single_sentence()
                EmbeddingWord2vec.learn(self, self.input_data, dim_number)
            elif self.concat_method == 'weighting' :
                EmbeddingWord2vec.extract_weighting(self, dim_number)

            matrix = self._reformat_matrix(self.index_ac, self.ac_weights)
            sup.create_file_from_list(
                matrix,
                os.path.join(self.embedded_path,
                             'ac_W2V_' + '{}_'.format(self.concat_method) + self.file_name.split('.')[0]  +'.emb'))
            return self.ac_weights

    def extract_log_info(self):

        self.activities, self.roles, self.times = [], [], []

        for log_case_id in self.log['caseid'].drop_duplicates():
            log_tmp = self.log[self.log['caseid'] == log_case_id]
            self.activities.append(list(log_tmp['task']))
            self.roles.append(list(log_tmp['user']))
            self.times.append(list(log_tmp['duration_cat']))


    def extract_weighting(self, dim_number):
        
        activities = self.learn_characteristics(self.activities, dim_number, 'activities')
        roles = self.learn_characteristics(self.roles, dim_number, 'roles')
        times = self.learn_characteristics(self.times, dim_number, 'times')

        count_roles = self.log[['task', 'user']].value_counts().reset_index().rename(columns={0: 'conteo'})
        count_duration = self.log[['task', 'duration_cat']].value_counts().reset_index().rename(columns={0: 'conteo'})

        embeddings_roles = {}
        for activity in self.ac_index.keys():
            count_roles_tmp = count_roles[count_roles['task']==activity]
            n = np.sum(count_roles_tmp['conteo'])
            
            role_vector = np.zeros((1, dim_number))
            for role in count_roles_tmp['user']:
                role_tmp = count_roles_tmp[count_roles_tmp['user']==role]
                user = role_tmp['user'].values[0]
                count = role_tmp['conteo'].values[0]
                role_vector = role_vector + (count/n)*roles[user]

            embeddings_roles[activity] = role_vector

        embeddings_times = {}
        for activity in self.ac_index.keys():
            count_duration_tmp = count_duration[count_duration['task']==activity]
            n = np.sum(count_duration_tmp['conteo'])
            
            time_vector = np.zeros((1, dim_number))
            for time in count_duration_tmp['duration_cat']:
                role_tmp = count_duration_tmp[count_duration_tmp['duration_cat']==time]
                duration_cat = role_tmp['duration_cat'].values[0]
                count = role_tmp['conteo'].values[0]
                time_vector = time_vector + (count/n)*times[duration_cat]

            embeddings_times[activity] = time_vector

        keys_sorted = [x[0]for x in sorted(self.ac_index.items(), key=lambda x: x[1], reverse=False)]

        for key in keys_sorted:
            if (embeddings_roles[key]==np.zeros((1, dim_number))).all():
                embeddings_roles[key] = np.ones((1, dim_number))
                
            if (embeddings_times[key]==np.zeros((1, dim_number))).all():
                embeddings_times[key] = np.ones((1, dim_number))
        
        if self.include_times:
            self.ac_weights = np.vstack([activities[key]*embeddings_roles[key]*embeddings_times[key] for key in keys_sorted])
        else:
            self.ac_weights = np.vstack([activities[key]*embeddings_roles[key] for key in keys_sorted])

    def extract_single_sentence(self):

        json_data = {}
        json_data['roles'] = self.roles
        json_data['activities'] = self.activities
        json_data['times'] = [list(map(str, y)) for y in self.times]

        if self.include_times:
            data = []
            for i in range(len(json_data['activities'])):
                tmp = [[x[0], x[1], x[2]] for x in list(zip(json_data['activities'][i], json_data['roles'][i], json_data['times'][i]))]
                data += tmp
        else:
            data = []
            for i in range(len(json_data['activities'])):
                tmp = [[x[0], x[1]] for x in list(zip(json_data['activities'][i], json_data['roles'][i], json_data['times'][i]))]
                data += tmp

        print(data[:10])
        self.input_data = data

    def extract_full_sentence(self):
        full_sentence = []
        if self.include_times:
            for log_case_id in self.log['caseid'].drop_duplicates():
                log_tmp = self.log[self.log['caseid'] == log_case_id]
                full_sentence.append([item for sublist in [[list(log_tmp['task'])[i], list(log_tmp['user'])[i], list(log_tmp['duration_cat'])[i]] for i in range(len(list(log_tmp['duration_cat'])))] for item in sublist])
        else:
            for log_case_id in self.log['caseid'].drop_duplicates():
                log_tmp = self.log[self.log['caseid'] == log_case_id]
                full_sentence.append([item for sublist in [[list(log_tmp['task'])[i], list(log_tmp['user'])[i]] for i in range(len(list(log_tmp['duration_cat'])))] for item in sublist])


        print(full_sentence[0])
        self.input_data = full_sentence

    def learn(self, sent, vectorsize):

        self.ac_weights = list()
        
        model = gensim.models.FastText(sent, vector_size = vectorsize,  min_count=0)
        
        nrEpochs = 100
        for epoch in range(nrEpochs):
            if epoch % 20 == 0:
                print('Now training epoch %s word2vec' % epoch)
            model.train(sent, start_alpha=0.025, epochs=nrEpochs, total_examples=model.corpus_count)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        keys_sorted = [x[0].replace(' ', '') for x in sorted(self.ac_index.items(), key=lambda x: x[1], reverse=False)]
        self.ac_weights = model.wv[keys_sorted]
        
    def learn_characteristics(self, sent, vectorsize, characteristic):

        self.ac_weights = list()
        model = gensim.models.FastText(sent, vector_size = vectorsize,  min_count=0, window=6)
    
        nrEpochs = 100
        for epoch in range(nrEpochs):
            if epoch % 20 == 0:
                print('Now training epoch %s word2vec' % epoch)
            model.train(sent, start_alpha=0.025, epochs=nrEpochs, total_examples=model.corpus_count)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        print(model.wv.key_to_index)
        if characteristic == 'activities':
            keys_sorted = [x[0] for x in sorted(self.ac_index.items(), key=lambda x: x[1], reverse=False)]
            chac_dict = {}
            for key in keys_sorted:
                chac_dict[key] = model.wv[key]
        else:
            unique_sent = list(set([item for sublist in sent for item in sublist]))
            chac_dict = {x:model.wv[x] for x in unique_sent}
        return chac_dict

    def _read_embedded(self, index, filename):
        """Loading of the embedded matrices.
        parms:
            index (dict): index of activities or roles.
            filename (str): filename of the matrix file.
        Returns:
            numpy array: array of weights.
        """
        weights = list()
        weights = pd.read_csv(os.path.join(self.embedded_path, filename),
                              header=None)
        weights[1] = weights.apply(lambda x: x[1].strip(), axis=1)
        if set(list(index.values())) == set(weights[1].tolist()):
            weights = weights.drop(columns=[0, 1])
            return np.array(weights)
        else:
            raise KeyError('Inconsistency in the number of activities')

    @staticmethod
    def _reformat_matrix(index, weigths):
            """Reformating of the embedded matrix for exporting.
            Args:
                index: index of activities or users.
                weigths: matrix of calculated coordinates.
            Returns:
                matrix with indexes.
            """
            matrix = list()
            for i, _ in enumerate(index):
                data = [i, index[i]]
                data.extend(weigths[i])
                matrix.append(data)
            return matrix
