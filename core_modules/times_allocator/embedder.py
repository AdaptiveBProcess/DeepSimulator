from core_modules.times_allocator import no_intercases_predictor as ndp
from core_modules.times_allocator import embedding_trainer as et
from core_modules.times_allocator import embedding_trainer_times as ett
from core_modules.times_allocator import embedding_trainer_act_weighting_times as etawt
from core_modules.times_allocator import embedding_trainer_act_weighting_no_times as etawnt
from core_modules.times_allocator import embedding_word2vec as ew
from core_modules.times_allocator import intercases_predictor_multimodel as mip
import os
import pandas as pd
import numpy as np


class Embedder():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, params, log, ac_index, index_ac, usr_index, index_usr):
        """constructor"""
        self.log = log.copy()
        self.params = params
        print(params)
        self.ac_index = ac_index
        self.index_ac = index_ac
        self.usr_index = usr_index
        self.index_usr = index_usr
        self.file_name = params['file']
        self.embedded_path = params['embedded_path']
        self.include_times = params['include_times']

    def Embedd(self, method):
        embedderclass = self._get_embedder(method)
        embedder = embedderclass(self.params, self.log, self.ac_index, self.index_ac, self.usr_index, self.index_usr)
        return embedder.load_embbedings()

    def _get_embedder(self, method):
        if method == 'emb_dot_product':
            return et.EmbeddingTrainer
        elif method == 'emb_w2vec':
            return ew.EmbeddingWord2vec
        elif method == 'emb_dot_product_times':
            return ett.EmbeddingTrainer
        elif method == 'emb_dot_product_act_weighting' and self.include_times:
            return etawt.EmbeddingTrainer
        elif method == 'emb_dot_product_act_weighting' and not self.include_times:
            return etawnt.EmbeddingTrainer
        else:
            raise ValueError(method)

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