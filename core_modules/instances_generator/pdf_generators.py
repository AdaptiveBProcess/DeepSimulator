# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:14:33 2020

@author: Manuel Camargo
"""
import numpy as np
import pandas as pd

from datetime import timedelta

from scipy.stats import expon, lognorm

from extraction import pdf_finder as pdf
import matplotlib.pyplot as plt


class PDFGenerator():
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, ia_times, ia_test):
        """constructor"""
        self.ia_times = ia_times
        self.ia_test = ia_test


    def generate(self, num_instances):
        dist = pdf.DistributionFinder(self.ia_times.inter_time.tolist()).distribution
        self.dist = dist
        print(self.dist)
        gen_inter = list()
        # TODO: Extender con mas distribuciones
        if self.dist['dname'] == 'EXPONENTIAL':
            gen_inter = expon.rvs(loc=0, scale=self.dist['dparams']['arg1'],
                            size=num_instances)
        if self.dist['dname'] == 'LOGNORMAL':
            m = self.dist['dparams']['mean']
            v = self.dist['dparams']['arg1']
            phi = np.sqrt(v + m**2)
            mu = np.log(m**2/phi)
            sigma = np.sqrt(np.log(phi**2/m**2))
            gen_inter = lognorm.rvs(sigma,
                                    scale=np.exp(mu),
                                    size=num_instances)
        now = self.ia_test.timestamp.min()
        times = list()
        for i, inter in enumerate(gen_inter):
            if i == 0:
                times.append(
                    {'caseid': 'Case'+str(i+1),
                      'timestamp': now + timedelta(seconds=inter)})
            else:
                times.append(
                    {'caseid': 'Case'+str(i+1),
                      'timestamp': (times[i-1]['timestamp'] +
                                    timedelta(seconds=inter))})
        self._graph_timeline(self.ia_test)
        self._graph_timeline(pd.DataFrame(times))

    @staticmethod
    def _graph_timeline(log) -> None:
        time_series = log.copy()[['caseid', 'timestamp']]
        time_series['occ'] = 1
        time_series.set_index('timestamp', inplace=True)
        time_series.occ.rolling('3h').sum().plot(figsize=(30,10), linewidth=5, fontsize=10)
        plt.xlabel('Days', fontsize=20);
        print(time_series)
