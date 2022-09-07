# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:23:55 2018

@author: Manuel Camargo
"""
import os
import random
import itertools
import math
import pandas as pd
import numpy as np
import utils.support as sup
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint




class EmbeddingTrainer():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        """Main method of the embedding training module.
        Args:
            parameters (dict): parameters for training the embeddeding network.
            timeformat (str): event-log date-time format.
            no_loops (boolean): remove loops fom the event-log (optional).
        """
        # Define the number of dimensions as the 4th root of the # of categories
        self.log = log.copy()
        self.ac_index = ac_index
        self.index_ac = index_ac
        self.usr_index = usr_index
        self.index_usr = index_usr
        self.file_name = params['file']
        self.embedded_path = params['embedded_path']


    def load_embbedings(self):
        # Embedded dimensions
        self.ac_weights = list()
        # Load embedded matrix
        ac_emb = 'ac_' + self.file_name.split('.')[0]+'.emb'
        if os.path.exists(os.path.join(self.embedded_path, ac_emb)):
            return self._read_embedded(self.index_ac, ac_emb)
        else:
            usr_idx = lambda x: self.usr_index.get(x['user'])
            self.log['usr_index'] = self.log.apply(usr_idx, axis=1)
                  
            dim_number = math.ceil(
                len(list(itertools.product(*[list(self.ac_index.items()),
                                             list(self.usr_index.items())])))**0.25)
            self._train_embedded(dim_number)
        
            if not os.path.exists(self.embedded_path):
                os.makedirs(self.embedded_path)
            print(type(self.ac_weights))
            print(type(self.index_ac))
            matrix = self._reformat_matrix(self.index_ac, self.ac_weights)
            sup.create_file_from_list(
                matrix,
                os.path.join(self.embedded_path,
                             'ac_' + self.file_name.split('.')[0]+'.emb'))
            return self.ac_weights


# =============================================================================
# Pre-processing: embedded dimension
# =============================================================================

    def _train_embedded(self, dim_number):
        """Carry out the training of the embeddings"""
        # Iterate through each book
        model = self._create_model(dim_number)
        model.summary()
    
        vec, cl = self._vectorize_input(self.log, negative_ratio=2)
        
        # Output file
        output_file_path = os.path.join(self.embedded_path,
                             self.file_name.split('.')[0]+
                                        '_emb.h5')
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
        ac_layer = model.get_layer('activity_embedding')
        self.ac_weights = ac_layer.get_weights()[0]
    
    def _vectorize_input(self, log, negative_ratio=1.0):
        """Generate batches of samples for training"""
        pairs = list()
        for i in range(0, len(self.log)):
            # Iterate through the links in the book


            pairs.append((self.ac_index[self.log.iloc[i]['task']],
                          self.usr_index[self.log.iloc[i]['user']]))
        print(len(self.log))

        print(self.ac_index)
        print(self.usr_index)
        print(self.log.processing_time)
        print(pairs)
        print(np.shape(pairs))
        n_positive = math.ceil(len(self.log)/2)
        batch_size = n_positive * (1 + negative_ratio)
        batch = np.zeros((batch_size, 3))
        pairs_set = set(pairs)
        activities = list(self.ac_index.keys())
        users = list(self.usr_index.keys())

        # This creates a generator
        # randomly choose positive examples
        idx = 0
        for idx, (activity, user) in enumerate(random.sample(pairs,
                                                             n_positive)):
            batch[idx, :] = (activity, user, 1)
        # Increment idx by 1
        idx += 1

        # Add negative examples until reach batch size
        while idx < batch_size:
            # random selection
            random_ac = random.randrange(len(activities)-1)
            random_rl = random.randrange(len(users)-1)

            # Check to make sure this is not a positive example
            if (random_ac, random_rl) not in pairs_set:

                # Add to batch and increment index,  0 due classification task
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        return {'activity': batch[:, 0], 'user': batch[:, 1]}, batch[:, 2]
        #     yield 
    
    def _create_model(self, embedding_size):
        """Model to embed activities and users using the functional API"""
    
        # Both inputs are 1-dimensional
        activity = Input(name='activity', shape=[1])
        user = Input(name='user', shape=[1])
       # times = Input(name='times', shape=[1])
    
        # Embedding the activity (shape will be (None, 1, embedding_size))
        activity_embedding = Embedding(name='activity_embedding',
                                       input_dim=len(self.ac_index),
                                       output_dim=embedding_size)(activity)
    
        # Embedding the user (shape will be (None, 1, embedding_size))
        user_embedding = Embedding(name='user_embedding',
                                   input_dim=len(self.usr_index),
                                   output_dim=embedding_size)(user)

        # Embedding the times (shape will be (None, 1, embedding_size))
        #times_embedding = Embedding(name='times_embedding',
                                   #input_dim=len(self.),
                                  # output_dim=embedding_size)(times)
    
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