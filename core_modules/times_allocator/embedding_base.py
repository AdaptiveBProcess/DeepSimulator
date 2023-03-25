import math
import random

from support_modules.common import EmbeddingMethods as Em
import gensim
import numpy as np


class EmbeddingBase:
    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        self.ac_weights = []
        self.log = log.copy()
        self.ac_index = ac_index
        self.index_ac = index_ac
        self.usr_index = usr_index
        self.index_usr = index_usr
        self.file_name = params['file']
        self.embedded_path = params['embedded_path']
        self.embedding_file_name = Em.get_matrix_file_name(
            params['emb_method'], params['include_times'], params['concat_method'], self.file_name)
        self.embedding_model_file_name = Em.get_model_file_name(params['emb_method'], params['include_times'], self.file_name)

    def learn_characteristics(self, sent, vector_size, characteristic):
        model = gensim.models.FastText(sent, vector_size=vector_size, min_count=0, window=6)

        nr_epochs = 100
        for epoch in range(nr_epochs):
            if epoch % 20 == 0:
                print('Now training epoch %s word2vec' % epoch)
            model.train(sent, start_alpha=0.025, epochs=nr_epochs, total_examples=model.corpus_count)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay

        print(model.wv.key_to_index)
        if characteristic == 'activities':
            keys_sorted = [x[0] for x in sorted(self.ac_index.items(), key=lambda x: x[1], reverse=False)]
            char_dict = {}
            for key in keys_sorted:
                char_dict[key] = model.wv[key]
        else:
            unique_sent = list(set([item for sublist in sent for item in sublist]))
            char_dict = {x: model.wv[x] for x in unique_sent}
        return char_dict

    def vectorize_input(self, log, negative_ratio=1.0):
        pairs = list()
        for i in range(0, len(self.log)):
            # Iterate through the links in the book
            pairs.append((self.ac_index[self.log.iloc[i]['task']],
                          self.usr_index[self.log.iloc[i]['user']]))

        n_positive = math.ceil(len(self.log) / 2)
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
            random_ac = random.randrange(len(activities) - 1)
            random_rl = random.randrange(len(users) - 1)

            # Check to make sure this is not a positive example
            if (random_ac, random_rl) not in pairs_set:
                # Add to batch and increment index,  0 due classification task
                batch[idx, :] = (random_ac, random_rl, 0)
                idx += 1

        # Make sure to shuffle order
        np.random.shuffle(batch)
        return {'activity': batch[:, 0], 'user': batch[:, 1]}, batch[:, 2]
