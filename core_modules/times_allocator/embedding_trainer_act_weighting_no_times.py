# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
import itertools
import math
import os
import numpy as np
import pandas as pd
import utils.support as sup
from support_modules.common import FileExtensions as Fe
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, Dot, Reshape
from keras.models import Model

from core_modules.times_allocator.embedding_base import EmbeddingBase


class EmbeddingTrainer(EmbeddingBase):
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        super().__init__(params, log, ac_index, index_ac, usr_index, index_usr)
        # Define the number of dimensions as the 4th root of the # of categories
        self.roles = []
        self.activities = []
        self.index_time = None
        self.time_index = None

    def load_embeddings(self):
        # Load embedded matrix
        if os.path.exists(os.path.join(self.embedded_path, self.embedding_file_name)):
            return self._read_embedded(self.index_ac, self.embedding_file_name)
        else:
            self.log['usr_index'] = self.log.apply(lambda x: self.usr_index.get(x['user']), axis=1)

            dim_number = math.ceil(
                len(list(itertools.product(*[list(self.ac_index.items()), list(self.usr_index.items())]))) ** 0.25)

            self.extract_log_info()
            weights = self.extract_weighting(dim_number)
            self._train_embedded(dim_number, weights)

            if not os.path.exists(self.embedded_path):
                os.makedirs(self.embedded_path)

            matrix = self._reformat_matrix(self.index_ac, self.ac_weights)
            sup.create_file_from_list(
                matrix,
                os.path.join(self.embedded_path, self.embedding_file_name))
            return self.ac_weights

    def extract_log_info(self):
        for log_case_id in self.log['caseid'].drop_duplicates():
            log_tmp = self.log[self.log['caseid'] == log_case_id]
            self.activities.append(list(log_tmp['task']))
            self.roles.append(list(log_tmp['user']))

    def extract_weighting(self, dim_number):

        activities = self.learn_characteristics(self.activities, dim_number, 'activities')
        roles = self.learn_characteristics(self.roles, dim_number, 'roles')

        count_roles = self.log[['task', 'user']].value_counts().reset_index().rename(columns={0: 'conteo'})

        embeddings_roles = {}
        for activity in self.ac_index.keys():
            count_roles_tmp = count_roles[count_roles['task'] == activity]
            n = np.sum(count_roles_tmp['conteo'])

            role_vector = np.zeros((1, dim_number))
            for role in count_roles_tmp['user']:
                role_tmp = count_roles_tmp[count_roles_tmp['user'] == role]
                user = role_tmp['user'].values[0]
                count = role_tmp['conteo'].values[0]
                role_vector = role_vector + (count / n) * roles[user]

            embeddings_roles[activity] = role_vector

        keys_sorted = [x[0] for x in sorted(self.ac_index.items(), key=lambda x: x[1], reverse=False)]

        for key in keys_sorted:
            if (embeddings_roles[key] == np.zeros((1, dim_number))).all():
                embeddings_roles[key] = np.ones((1, dim_number))

        return np.vstack([activities[key] * embeddings_roles[key] for key in keys_sorted])

    # =============================================================================
    # Pre-processing: embedded dimension
    # =============================================================================

    def _train_embedded(self, dim_number, act_weights):
        """Carry out the training of the embeddings"""
        # Iterate through each book
        model = self._create_model(dim_number, act_weights)
        model.summary()

        vec, cl = self.vectorize_input(self.log, negative_ratio=2)

        # Output file
        output_file_path = os.path.join(self.embedded_path, self.embedding_model_file_name)
        # Saving
        model_checkpoint = ModelCheckpoint(output_file_path,
                                           monitor='val_loss',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto')
        # Train
        print(vec)
        model.fit(x=vec, y=cl,
                  validation_split=0.2,
                  callbacks=[model_checkpoint],
                  epochs=100,
                  verbose=2)

        # Extract embeddings
        ac_layer = model.get_layer('activity_embedding')
        self.ac_weights = ac_layer.get_weights()[0]

    def _create_model(self, embedding_size, act_weights):
        """Model to embed activities and users using the functional API"""

        # Both inputs are 1-dimensional
        activity = Input(name='activity', shape=[1])
        user = Input(name='user', shape=[1])

        # Poner matriz de weights en la entrada con la salida de los weighting embeddings
        # Embedding the activity (shape will be (None, 1, embedding_size))
        activity_embedding = Embedding(act_weights.shape[0],
                                       act_weights.shape[1],
                                       weights=[act_weights],
                                       name='activity_embedding')(activity)

        # Embedding the user (shape will be (None, 1, embedding_size))
        user_embedding = Embedding(name='user_embedding',
                                   input_dim=len(self.usr_index),
                                   output_dim=embedding_size)(user)

        # Merge the layers with a dot product
        # along the second axis (shape will be (None, 1, 1))
        # merged = Multiply(name='dot_product')([activity_embedding, user_embedding, time_embedding])

        merged = Dot(name='dot_product_act_usr',
                     normalize=True, axes=2)([activity_embedding, user_embedding])

        # Reshape to be a single number (shape will be (None, 1))
        merged = Reshape(target_shape=[1])(merged)

        # Loss function is mean squared error
        model = Model(inputs=[activity, user], outputs=merged)
        model.compile(optimizer='Adam', loss='mse')

        return model

    # =============================================================================
    # Support
    # =============================================================================

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

    def _read_embedded(self, index, filename):
        """Loading of the embedded matrices.
        params:
            index (dict): index of activities or roles.
            filename (str): filename of the matrix file.
        Returns:
            numpy array: array of weights.
        """
        weights = pd.read_csv(os.path.join(self.embedded_path, filename), header=None)
        weights[1] = weights.apply(lambda x: x[1].strip(), axis=1)
        if set(list(index.values())) == set(weights[1].tolist()):
            weights = weights.drop(columns=[0, 1])
            return np.array(weights)
        else:
            raise KeyError('Inconsistency in the number of activities')
