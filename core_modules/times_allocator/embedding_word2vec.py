# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:40:36 2021

@author: Manuel Camargo
"""

# -*- coding: utf-8 -*-
from support_modules import log_reader as lr
# from support_modules import role_discovery as rd
import gensim
import os
import pandas as pd
import numpy as np


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

    def load_embbedings(self):
        # Embedded dimensions
        self.ac_weights = list()
        # Load embedded matrix
        ac_emb = 'ac_' + self.file_name.split('.')[0] + '.emb'
        if os.path.exists(os.path.join(self.embedded_path, ac_emb)):
            return self._read_embedded(self.index_ac, ac_emb)
        else:
            activities = lr.get_sentences_XES(self.file_name)
            EmbeddingWord2vec.learn(self,activities,4)

    def learn(self,sent, vectorsize):
        # train model
        model = gensim.models.Word2Vec(sent, vector_size = vectorsize,  min_count=0)
        nrEpochs = 10
        for epoch in range(nrEpochs):
            if epoch % 2 == 0:
                print('Now training epoch %s' % epoch)
            model.train(sent, start_alpha=0.025, epochs=nrEpochs, total_examples=model.corpus_count)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        # print(model.wv.most_similar('AnalyzePurchaseRequisition'))
        # print(model.wv['AmendPurchaseRequisition'])
        # print(model.wv['AmendRequestforQuotation'])
        # print(model.wv['AnalyzePurchaseRequisition'])
        # print(model.wv['AnalyzeQuotationComparisonMap'])
        # print(model.wv['AnalyzeRequestforQuotation'])
        # print(model.wv['ApprovePurchaseOrderforpayment'])
        # print(model.wv["AuthorizeSupplier'sInvoicepayment"])
        # print(model.wv['Choosebestoption'])
        # print(model.wv['ConfirmPurchaseOrder'])
        # print(model.wv['CreatePurchaseOrder'])
        # print(model.wv['CreatePurchaseRequisition'])
        # print(model.wv['CreateQuotationcomparisonMap'])
        # print(model.wv['CreateRequestforQuotation'])
        # print(model.wv['DeliverGoodsServices'])
        # print(model.wv['PayInvoice'])
        # print(model.wv['ReleasePurchaseOrder'])
        # print(model.wv["ReleaseSupplier'sInvoice"])
        # print(model.wv['SendInvoice'])
        # print(model.wv['SendRequestforQuotationtoSupplier'])
        # print(model.wv['SettleConditionsWithSupplier'])
        # print(model.wv['SettleDisputeWithSupplier'])

        model.wv.save(self.embedded_path + '/A2VVS' + str(vectorsize) + '.model')

    def learnResources(rol, vectorsize):
        # train model
        model = gensim.models.Word2Vec(rol, vectorsize, window=3, min_count=0)
        nrEpochs = 10
        for epoch in range(nrEpochs):
            if epoch % 2 == 0:
                print('Now training epoch %s' % epoch)
            model.train(rol, start_alpha=0.025, epochs=nrEpochs, total_examples=model.corpus_count)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        # print(model.wv.most_similar('PedroAlvares'))
        # print(model.wv['PedroAlvares'])
        # print(model.wv['KaraldaNimwada'])
        # print(model.wv['KiuKan'])
        # print(model.wv['EsmeraldaClay'])
        # print(model.wv['CarmenFinacse'])
        # print(model.wv['SeanManney'])
        # print(model.wv['KarenClarens'])
        # print(model.wv['ElviraLores'])
        # print(model.wv['KimPassa'])
        # print(model.wv['AnnaKaufmann'])
        # print(model.wv['FjodorKowalski'])
        model.wv.save('/content/drive/MyDrive/MBIT/DeepLearning/' + 'A2VVS' + str(vectorsize) + '.model')

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