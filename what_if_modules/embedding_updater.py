# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:26:44 2021

@author: Manuel Camargo
"""
import os
import numpy as np
import pandas as pd
import json
import math
import random

import utils.support as sup

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Dot, Reshape


class EmbeddingUpdater():
    """
    """

    def __init__(self, parms):
        self.new_ac_weights = None
        self.modif_model_path = os.path.join(
            parms['gl']['embedded_path'],
            parms['gl']['modified_file'].split('.')[0] + '_emb.h5')
        self.modif_embedding_path = os.path.join(
            parms['gl']['embedded_path'],
            'ac_' + parms['gl']['modified_file'].split('.')[0] + '.emb')
        self.modif_gen_metadata_path = os.path.join(
            parms['gl']['times_gen_path'],
            parms['gl']['modified_file'].split('.')[0] + '_diapr_meta.json')
        self.complete_gen_metadata_path = os.path.join(
            parms['gl']['times_gen_path'],
            parms['gl']['complete_file'].split('.')[0] + '_diapr_meta.json')
        self.modif_params = dict()

        self.output_file_path = os.path.join(
            parms['gl']['embedded_path'],
            'ac_' + parms['gl']['modified_file'].split('.')[0] + '_upd.emb')
        self.output_model_path = os.path.join(
            parms['gl']['embedded_path'],
            parms['gl']['modified_file'].split('.')[0] + '_upd_emb.h5')

    def execute_pipeline(self):
        model, ac_weights, org_parms = self._read_original_model()
        self.define_modif_parms(org_parms)
        print(self.modif_params['m_tasks'])
        # extend embedding
        new_ac_weights = np.append(
            ac_weights,
            np.random.rand(len(self.modif_params['m_tasks']),
                           ac_weights.shape[1]), axis=0)
        # Create new model
        output_shape = model.get_layer('activity_embedding').output_shape
        users_input_dim = model.get_layer('user_embedding').input_dim
        num_emb_dimmensions = output_shape[2]
        new_model = self.create_embedding_model(new_ac_weights.shape[0], users_input_dim, num_emb_dimmensions)
        # Save weights of old model
        temp_weights = {layer.name: layer.get_weights() for layer in model.layers}
        # load values in new model
        for k, v in temp_weights.items():
            if k == 'activity_embedding':
                new_model.get_layer(k).set_weights([new_ac_weights])
            elif v:
                new_model.get_layer(k).set_weights(v)
        # Re-train embedded model and update dimmensions
        self.new_ac_weights = self.re_train_embedded(new_model, num_emb_dimmensions)
        # Save results
        matrix = self.reformat_matrix(
            {v: k for k, v in org_parms['ac_index'].items()}, new_ac_weights)
        sup.create_file_from_list(matrix, self.output_file_path)

    def _read_original_model(self):
        # Load old model
        model = load_model(self.modif_model_path)
        model.summary()
        # Load embeddings    
        ac_weights = self.read_embedded(self.modif_embedding_path)
        # Read data
        with open(self.modif_gen_metadata_path) as file:
            parameters = json.load(file)
        return model, ac_weights, parameters

    def define_modif_parms(self, org_parms):
        with open(self.complete_gen_metadata_path) as file:
            exp_parms = json.load(file)
        # ac and roles missings comparison
        m_tasks = list(set(
            exp_parms['ac_index'].keys()) - set(org_parms['ac_index'].keys()))
        m_tasks_assignment = list(
            filter(lambda x: x['task'] in m_tasks, exp_parms['roles_table']))
        # Save params
        self.modif_params['ac_index'] = org_parms['ac_index']
        self.modif_params['usr_index'] = org_parms['usr_index']
        self.modif_params['m_tasks'] = m_tasks
        self.modif_params['m_tasks_assignment'] = m_tasks_assignment
        self.modif_params['roles'] = exp_parms['roles']
        self.modif_params['len_log'] = org_parms['log_size']
        # extend indexes
        for task in m_tasks:
            self.modif_params['ac_index'][task] = len(
                self.modif_params['ac_index'])

    def create_embedding_model(self, num_cat, num_users, embedding_size):
        """Model to embed activities and users using the functional API"""

        # Both inputs are 1-dimensional
        activity = Input(name='activity', shape=[1])
        user = Input(name='user', shape=[1])

        # Embedding the activity (shape will be (None, 1, embedding_size))
        activity_embedding = Embedding(name='activity_embedding',
                                       input_dim=num_cat,
                                       output_dim=embedding_size)(activity)

        # Embedding the user (shape will be (None, 1, embedding_size))
        user_embedding = Embedding(name='user_embedding',
                                   input_dim=num_users,
                                   output_dim=embedding_size)(user)

        # Merge the layers with a dot product
        # along the second axis (shape will be (None, 1, 1))
        merged = Dot(name='dot_product',
                     normalize=True, axes=2)([activity_embedding, user_embedding])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape=[1])(merged)

        # Loss function is mean squared error
        model = Model(inputs=[activity, user], outputs=merged)
        model.compile(optimizer='Adam', loss='mse')

        return model

    def re_train_embedded(self, model, dim_number):
        """Carry out the training of the embeddings"""
        # Iterate through each book
        vec, cl = self.vectorize_input(self.modif_params, negative_ratio=2)

        # Output file
        output_file_path = os.path.join(self.output_model_path)
        # Saving
        model_checkpoint = ModelCheckpoint(output_file_path,
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto')
        # Train
        model.fit(x=vec, y=cl,
                  validation_split=0.2,
                  callbacks=[model_checkpoint],
                  epochs=100,
                  verbose=2)

        # Extract embeddings
        return model.get_layer('activity_embedding').get_weights()[0]

    @staticmethod
    def vectorize_input(modif_params, negative_ratio=1.0):
        """Generate batches of samples for training"""
        roles = (pd.DataFrame.from_dict(modif_params['roles'], orient='index')
                 .unstack()
                 .reset_index()
                 .drop(columns='level_0')
                 .rename(columns={'level_1': 'role', 0: 'user'}))
        roles = roles[~roles.user.isna()]
        # Task assignments
        m_tasks_assignment = pd.DataFrame(modif_params['m_tasks_assignment'])
        roles = roles.merge(m_tasks_assignment, on='role', how='left')
        pairs = roles[['task', 'user']][~roles.task.isna()].to_records(index=False)
        pairs = [(modif_params['ac_index'][x[0]], modif_params['usr_index'][x[1]]) for x in pairs]

        n_positive = math.ceil(modif_params['len_log'] * 0.15)
        batch_size = n_positive * (1 + negative_ratio)
        batch = np.zeros((batch_size, 3))
        pairs_set = set(pairs)
        activities = modif_params['m_tasks']
        users = list(modif_params['usr_index'].keys())
        # This creates a generator
        # randomly choose positive examples
        idx = 0
        for idx in range(n_positive):
            activity, user = random.sample(pairs, 1)[0]
            batch[idx, :] = (activity, user, 1)
        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:
            # random selection
            random_ac = modif_params['ac_index'][random.sample(activities, 1)[0]]
            random_rl = random.randrange(len(users) - 1)

            # Check to make sure this is not a positive example
            if (random_ac, random_rl) not in pairs_set:
                # Add to batch and increment index,  0 due classification task
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1
        # Make sure to shuffle order
        np.random.shuffle(batch)
        return {'activity': batch[:, 0], 'user': batch[:, 1]}, batch[:, 2]

    # =============================================================================
    #   Support modules
    # =============================================================================
    @staticmethod
    def read_embedded(filename):
        """Loading of the embedded matrices.
        parms:
            index (dict): index of activities or roles.
            filename (str): filename of the matrix file.
        Returns:
            numpy array: array of weights.
        """
        weights = pd.read_csv(filename, header=None)
        weights[1] = weights.apply(lambda x: x[1].strip(), axis=1)
        weights = weights.drop(columns=[0, 1])
        return weights.to_numpy()

    @staticmethod
    def reformat_matrix(index, weigths):
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
