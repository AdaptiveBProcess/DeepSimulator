# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:40:36 2021

@author: Manuel Camargo
"""
from core_modules.times_allocator import no_intercases_predictor as ndp
from core_modules.times_allocator import intercases_predictor as ip
from core_modules.times_allocator import intercases_predictor_multimodel as mip


class TimesPredictor:
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, model_path, parms, sequences, iarr):
        """constructor"""
        self.model_path = model_path
        self.parms = parms
        self.sequences = sequences
        self.iarr = iarr
        
    def predict(self, method):
        predictor = self._get_predictor(method)
        return predictor.predict(self.sequences, self.iarr)

    def _get_predictor(self, method):
        if method == 'basic':
            return ndp.NoIntercasesPredictor(self.model_path, self.parms)
        elif method in ['inter', 'inter_nt']:
            return ip.IntercasesPredictor(self.model_path, self.parms)
        elif method == 'dual_inter':
            return mip.DualIntercasesPredictor(self.model_path, self.parms)
        else:
            raise ValueError(method)