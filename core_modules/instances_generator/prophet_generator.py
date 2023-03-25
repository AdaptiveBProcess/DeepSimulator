# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:21:50 2021

@author: Manuel Camargo
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
import shutil
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
import itertools
import traceback

import utils.support as sup
from support_modules.common import LogAttributes as La
from support_modules.common import FileExtensions as Fe

import logging

logger = logging.getLogger('fbprophet.plot')
logger.setLevel(logging.CRITICAL)
from fbprophet import Prophet
from fbprophet.serialize import model_to_json, model_from_json
from fbprophet.diagnostics import cross_validation, performance_metrics

from numpy.random import triangular as triang

import matplotlib.pyplot as plt


class ProphetGenerator:
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, log, valdn, parms):
        """constructor"""
        self.times = None
        self.temp_output = os.path.join('output_files', sup.folder_id())
        if not os.path.exists(self.temp_output):
            os.makedirs(self.temp_output)
        self.log = pd.DataFrame(log.data)
        self.valdn = valdn
        self.parms = parms
        self.model_metadata = dict()
        self.is_safe = True
        self._load_model()

    # @safe_exec
    def _load_model(self) -> None:
        model_path = os.path.join(self.parms['ia_gen_path'], f"{self.parms['file'].split('.')[0]}_prf{Fe.JSON}")
        # Save path(s) if the model exists 
        self.model_path = model_path
        model_exist = os.path.exists(model_path)
        self.parms['model_path'] = model_path
        # Discover and compare
        if not model_exist or self.parms['update_ia_gen']:
            acc = self._discover_model()
            save, metadata_file = self._compare_models(acc, self.parms['update_ia_gen'])
            if save:
                self._save_model(metadata_file, acc)
            else:
                shutil.rmtree(self.temp_output)

    def _compare_models(self, acc, model_exist):
        metadata_file = os.path.join(self.parms['ia_gen_path'], f"{self.parms['file'].split('.')[0]}_prf_meta{Fe.JSON}")
        # compare with existing model
        save = True
        if model_exist:
            # Loading of parameters from existing model
            if os.path.exists(metadata_file):
                with open(metadata_file) as file:
                    data = json.load(file)
                if data['loss'] < acc['loss']:
                    save = False
        return save, metadata_file

    # @safe_exec
    def _discover_model(self):
        df_train, self.max_cap = self._transform_features(self.log)
        # print(df_train)
        days = df_train.ds.max() - df_train.iloc[int(len(df_train) * 0.8)].ds
        periods = days * 0.5

        param_grid = {'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                      'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]}

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []  # Store the RMSEs for each params here

        # Use cross validation to evaluate all parameters
        for params in all_params:
            params['growth'] = 'logistic'
            try:
                m = Prophet(**params).fit(df_train)  # Fit model with given params
                df_cv = cross_validation(m, horizon=days, period=periods, parallel="processes")
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])
            except:
                # df_train.to_csv('df_train_fail.csv')
                traceback.print_exc()
                pass

        # Find the best parameters
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        best_params = all_params[np.argmin(rmses)]
        m = Prophet(**best_params).fit(df_train)
        with open(os.path.join(self.temp_output, f"{self.parms['file'].split('.')[0]}_prf{Fe.JSON}"), 'w') as fout:
            json.dump(model_to_json(m), fout, cls=NumpyEncoder)  # Save model
        return {'loss': tuning_results.iloc[np.argmin(rmses)].rmse}

    def _save_model(self, metadata_file, acc):
        # best structure mining parameters
        self.model_metadata['loss'] = acc['loss']
        self.model_metadata['generated_at'] = (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        self.model_metadata['max_cap'] = self.max_cap

        # Copy best model to destination folder
        destintion = os.path.join(self.parms['ia_gen_path'], f"{self.parms['file'].split('.')[0]}_prf{Fe.JSON}")
        source = os.path.join(self.temp_output, f"{self.parms['file'].split('.')[0]}_prf{Fe.JSON}")

        shutil.copyfile(source, destintion)
        # Save metadata
        sup.create_json(self.model_metadata, metadata_file)
        # clean output folder
        shutil.rmtree(self.temp_output)

    @staticmethod
    def _transform_features(df_train):
        df_train = df_train.groupby(La.CASE_ID).start_timestamp.min().reset_index()
        df_train = df_train.groupby([pd.Grouper(key=La.START_TIME, freq='H')]).size().reset_index(name='count')
        df_train.rename(columns={La.START_TIME: 'ds', 'count': 'y'}, inplace=True)
        df_train = df_train.fillna(1)

        max_cap = df_train.y.max() * 1.2
        df_train['cap'] = max_cap
        df_train['floor'] = 0
        df_train['y'] = df_train['y'].astype(np.float64)
        df_train['cap'] = df_train['cap'].astype(np.float64)
        df_train['floor'] = df_train['floor'].astype(np.float64)
        print(df_train.dtypes)
        return df_train, max_cap

    # @safe_exec
    def generate(self, num_instances, start_time):
        with open(self.model_path, 'r') as fin:
            m = model_from_json(json.load(fin))  # Load model
        metadata_file = os.path.join(self.parms['ia_gen_path'], f"{self.parms['file'].split('.')[0]}_prf_meta{Fe.JSON}")
        # compare with existing model
        with open(metadata_file) as file:
            max_cap = json.load(file)['max_cap']
        # times
        start_time = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S.%f+00:00")
        gen = list()
        n_gen_inst = 0
        while n_gen_inst < num_instances:
            future = pd.date_range(start=start_time, end=(start_time + timedelta(days=30)), freq='H').to_frame(
                name='ds', index=False)

            future['cap'] = max_cap
            future['floor'] = 0
            forecast = m.predict(future)

            def rand_value(x):
                raw_val = triang(x.yhat_lower, x.yhat, x.yhat_upper, size=1)
                raw_val = raw_val[0] if raw_val[0] > 0 else 0
                return raw_val

            forecast['gen'] = forecast.apply(rand_value, axis=1)
            forecast['gen_round'] = np.ceil(forecast['gen'])
            n_gen_inst += np.sum(forecast['gen_round'])
            gen.append(forecast[forecast.gen_round > 0][['ds', 'gen_round']])
            start_time = forecast.ds.max()
        gen = pd.concat(gen, axis=0, ignore_index=True)

        def pp(start, n):
            start_u = int(start.value // 10 ** 9)
            end_u = int((start + timedelta(hours=1)).value // 10 ** 9)
            return pd.to_datetime(np.random.randint(start_u, end_u, int(n)), unit='s').to_frame(name=La.TIMESTAMP)

        gen_cases = list()
        for row in gen.itertuples(index=False):
            gen_cases.append(pp(row.ds, row.gen_round))
        self.times = pd.concat(gen_cases, axis=0, ignore_index=True)
        self.times = self.times.iloc[:num_instances]
        self.times[La.CASE_ID] = self.times.index + 1
        self.times[La.CASE_ID] = self.times[La.CASE_ID].astype(str)
        self.times[La.CASE_ID] = 'Case' + self.times[La.CASE_ID]
        return self.times

    @staticmethod
    def _graph_timeline(log) -> None:
        time_series = log.copy()[[La.CASE_ID, La.TIMESTAMP]]
        time_series['occ'] = 1
        time_series.set_index(La.TIMESTAMP, inplace=True)
        time_series.occ.rolling('3h').sum().plot(figsize=(30, 10), linewidth=5, fontsize=10)
        plt.xlabel('Days', fontsize=20)


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.bool_):
            return bool(obj)

        elif isinstance(obj, np.void):
            return None
        return json.JSONEncoder.default(self, obj)
