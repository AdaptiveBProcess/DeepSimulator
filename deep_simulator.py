# -*- coding: utf-8 -*-
"""
@author: Manuel Camargo
"""
import copy
import os
import shutil
import warnings
from operator import itemgetter

import analyzers.sim_evaluator as sim
import pandas as pd
import readers.bpmn_reader as br
import readers.log_reader as lr
import readers.process_structure as gph
import utils.support as sup
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA, TruncatedSVD, DictionaryLearning
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from utils.support import timeit, safe_exec

import support_modules.common as cm
from core_modules.instances_generator import instances_generator as gen
from core_modules.sequences_generator import seq_generator as sg
from core_modules.times_allocator import times_generator as ta

warnings.filterwarnings("ignore")


class DeepSimulator:
    """
    Main class of the Simulator
    """

    def __init__(self, parms):
        """constructor"""
        self.parms = parms
        self.is_safe = True
        self.sim_values = list()

    def execute_pipeline(self) -> None:
        exec_times = dict()
        self.is_safe = self._read_inputs(log_time=exec_times, is_safe=self.is_safe)
        # get minimum date
        start_time = (self.log_test.start_timestamp.min().strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"))
        print('############ Structure optimization ############')
        # Structure optimization
        seq_generator_class = sg.SeqGeneratorFabric.get_generator(self.parms['s_gen']['gen_method'])
        seq_gen = seq_generator_class({**self.parms['gl'], **self.parms['s_gen']}, self.log_train)
        print('############ Generate inter-arrivals ############')
        self.is_safe = self._read_bpmn(log_time=exec_times, is_safe=self.is_safe)
        generator = gen.InstancesGenerator(self.process_graph, self.log_train, self.parms['i_gen']['gen_method'],
                                           {**self.parms['gl'], **self.parms['i_gen']})
        print('########### Generate instances times ###########')
        times_allocator = ta.TimesGenerator(self.process_graph, self.log_train,
                                            {**self.parms['gl'], **self.parms['t_gen']})
        output_path = os.path.join('output_files', sup.folder_id())
        for rep_num in range(0, self.parms['gl']['exp_reps']):
            seq_gen.generate(self.log_test, start_time)
            if self.parms['i_gen']['gen_method'] == 'test':
                inter_arrival = generator.generate(seq_gen.gen_seqs, start_time)
            else:
                inter_arrival = generator.generate(len(self.log_test.caseid.unique()), start_time)
            seq_gen.clean_time_stamps()
            event_log = times_allocator.generate(seq_gen.gen_seqs, inter_arrival)
            event_log = pd.DataFrame(event_log)
            # Export log
            self._export_log(event_log, output_path, rep_num)
            # Evaluate log
            if self.parms['gl']['evaluate']:
                self.sim_values.extend(self._evaluate_logs(self.parms, self.log_test, event_log, rep_num))
        self._export_results(output_path)
        print("-- End of trial --")

    @timeit
    @safe_exec
    def _read_inputs(self, **_kwargs) -> None:
        # Event log reading
        self.log = lr.LogReader(os.path.join(self.parms['gl']['event_logs_path'], self.parms['gl']['file']),
                                self.parms['gl']['read_options'])
        # Time splitting 80-20
        self._split_timeline(0.8, self.parms['gl']['read_options']['one_timestamp'])

    @timeit
    @safe_exec
    def _read_bpmn(self, **_kwargs) -> None:
        bpmn_path = os.path.join(self.parms['gl']['bpmn_models'], self.parms['gl']['file'].split('.')[0] + '.bpmn')
        self.bpmn = br.BpmnReader(bpmn_path)
        self.process_graph = gph.create_process_structure(self.bpmn)

    @staticmethod
    def _evaluate_logs(parms, log, sim_log, rep_num):
        """Reads the simulation results stats
        Args:
            settings (dict): Path to jar and file names
            rep (int): repetition number
        """
        sim_values = list()
        log = copy.deepcopy(log)
        log = log[~log.task.isin(['Start', 'End'])]
        log['source'] = 'log'
        log.rename(columns={'user': 'resource'}, inplace=True)
        log['caseid'] = log['caseid'].astype(str)
        log['caseid'] = 'Case' + log['caseid']
        evaluator = sim.SimilarityEvaluator(log, sim_log, parms['gl'], max_cases=1000)
        metrics = [parms['gl']['sim_metric']]
        if 'add_metrics' in parms['gl'].keys():
            metrics = list(set(list(parms['gl']['add_metrics']) + metrics))
        for metric in metrics:
            evaluator.measure_distance(metric)
            sim_values.append({**{'run_num': rep_num}, **evaluator.similarity})
        return sim_values

    def _export_log(self, event_log, output_path, r_num) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        event_log.to_csv(
            os.path.join(output_path, 'gen_' + self.parms['gl']['file'].split('.')[0] + '_' + str(r_num + 1) + '.csv'),
            index=False)

    @staticmethod
    def clustering_method(dataframe, method, k=3):

        cols = [x for x in dataframe.columns if 'id_' in x]
        x = dataframe[cols]

        if method == 'kmeans':
            kmeans = KMeans(n_clusters=k, random_state=30).fit(x)
            dataframe['cluster'] = kmeans.labels_
        elif method == 'mean_shift':
            ms = MeanShift(bandwidth=k, bin_seeding=True).fit(x)
            dataframe['cluster'] = ms.labels_
        elif method == 'gaussian_mixture':
            dataframe['cluster'] = GaussianMixture(
                n_components=k, covariance_type='spherical', random_state=30).fit_predict(x)

        return dataframe

    @staticmethod
    def decomposition_method(dataframe, method):

        cols = [x for x in dataframe.columns if 'id_' in x]
        X = dataframe[cols]

        if method == 'pca':
            dataframe[['x', 'y', 'z']] = PCA(n_components=3).fit_transform(X)
        elif method == 'truncated_svd':
            dataframe[['x', 'y', 'z']] = TruncatedSVD(n_components=3).fit_transform(X)
        elif method == 'dictionary_learning':
            dataframe[['x', 'y', 'z']] = DictionaryLearning(n_components=3,
                                                            transform_algorithm='lasso_lars').fit_transform(X)

        return dataframe

    def _clustering_metrics(self, params):

        file_name = params['gl']['file']
        embedded_path = params['gl']['embedded_path']
        concat_method = params['t_gen']['concat_method']
        include_times = params['t_gen']['include_times']

        if params['t_gen']['emb_method'] == 'emb_dot_product':
            emb_path = os.path.join(embedded_path, 'ac_DP_' + file_name.split('.')[0] + '.emb')
        elif params['t_gen']['emb_method'] == 'emb_w2vec':
            emb_path = os.path.join(embedded_path,
                                    'ac_W2V_' + '{}_'.format(concat_method) + file_name.split('.')[0] + '.emb')
        elif params['t_gen']['emb_method'] == 'emb_dot_product_times':
            emb_path = os.path.join(embedded_path, 'ac_DP_times_' + file_name.split('.')[0] + '.emb')
        elif params['t_gen']['emb_method'] == 'emb_dot_product_act_weighting' and include_times:
            emb_path = os.path.join(embedded_path, 'ac_DP_act_weighting_times_' + file_name.split('.')[0] + '.emb')
        elif params['t_gen']['emb_method'] == 'emb_dot_product_act_weighting' and not include_times:
            emb_path = os.path.join(embedded_path, 'ac_DP_act_weighting_no_times_' + file_name.split('.')[0] + '.emb')

        df_embeddings = pd.read_csv(emb_path, header=None)
        n_cols = len(df_embeddings.columns)
        df_embeddings.columns = ['id', 'task_name'] + ['id_{}'.format(idx) for idx in range(1, n_cols - 1)]
        df_embeddings['task_name'] = df_embeddings['task_name'].str.lstrip()

        """
        clustering_ms = ['kmeans', 'gaussian_mixture']
        decomposition_ms = ['pca', 'truncated_svd']
        KS = [3, 5, 7]
        """

        clustering_ms = ['kmeans']
        decomposition_ms = ['pca']
        KS = [3]

        metrics = []
        for clustering_m in clustering_ms:
            for decomposition_m in decomposition_ms:
                for K in KS:
                    df_embeddings_tmp = self.clustering_method(df_embeddings, clustering_m, K)
                    df_embeddings_tmp = self.decomposition_method(df_embeddings_tmp, decomposition_m)
                    s_score = silhouette_score(df_embeddings_tmp[['x', 'y', 'z']], df_embeddings_tmp['cluster'],
                                               metric='euclidean')
                    ch_score = calinski_harabasz_score(df_embeddings_tmp[['x', 'y', 'z']], df_embeddings_tmp['cluster'])
                    metrics.append([clustering_m, decomposition_m, K, s_score, ch_score])

        metrics_df = pd.DataFrame(data=metrics, columns=['clustering_method', 'decomposition_method', 'number_clusters',
                                                         'silhouette_score', 'calinski_harabasz_score'])
        best = metrics_df.sort_values(by=['silhouette_score', 'calinski_harabasz_score'], ascending=True).head(1)

        return best.T.reset_index()

    def _export_results(self, output_path) -> None:

        clust_mets = self._clustering_metrics(self.parms)
        clust_mets.columns = ['metric', 'sim_val']
        clust_mets['run_num'] = 0.0
        sim_values_df = pd.DataFrame(self.sim_values).sort_values(by='metric')
        results_df = pd.concat([sim_values_df, clust_mets])

        self._save_embedding_metrics_results(output_path, results_df)

        # Save logs        
        log_test = self.log_test[~self.log_test.task.isin(['Start', 'End'])]
        log_test.to_csv(os.path.join(output_path, 'tst_' + self.parms['gl']['file'].split('.')[0] + '.csv'),
                        index=False)
        if self.parms['gl']['save_models']:
            paths = ['bpmn_models', 'embedded_path', 'ia_gen_path', 'seq_flow_gen_path', 'times_gen_path']
            sources = list()
            for path in paths:
                for root, dirs, files in os.walk(self.parms['gl'][path]):
                    for file in files:
                        if self.parms['gl']['file'].split('.')[0] in file:
                            sources.append(os.path.join(root, file))
            for source in sources:
                base_folder = os.path.join(output_path, os.path.basename(os.path.dirname(source)))
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)
                destination = os.path.join(base_folder, os.path.basename(source))
                # Copy dl models
                allowed_ext = self._define_model_path({**self.parms['gl'], **self.parms['t_gen']})
                is_dual = self.parms['t_gen']['model_type'] == 'dual_inter'
                if is_dual and ('times_gen_models' in source) and any([x in source for x in allowed_ext]):
                    shutil.copyfile(source, destination)
                elif not is_dual and ('times_gen_models' in source) and any(
                        [self.parms['gl']['file'].split('.')[0] + x in source for x in allowed_ext]):
                    shutil.copyfile(source, destination)
                # copy other models
                folders = ['bpmn_models', 'embedded_matix', 'ia_gen_models']
                allowed_ext = ['.emb', '.bpmn', '_mpdf.json', '_prf.json', '_prf_meta.json', '_mpdf_meta.json',
                               '_meta.json']
                if any([x in source for x in folders]) and any(
                        [self.parms['gl']['file'].split('.')[0] + x in source for x in allowed_ext]):
                    shutil.copyfile(source, destination)

    def _save_embedding_metrics_results(self, output_path, results_df):
        # Save results
        results_df.to_csv(os.path.join(output_path, sup.file_id(prefix='SE_')), index=False)
        results_df_t = results_df.set_index('metric').T.reset_index(drop=True)
        input_method, include_times = cm.EmbeddingMethods.get_input_and_times_method(
            self.parms['t_gen']['emb_method'], self.parms['t_gen']['include_times'],
            self.parms['t_gen']['concat_method'])
        results_df_t['input_method'] = input_method
        results_df_t['embedding_method'] = cm.EmbeddingMethods.get_base_model(self.parms['t_gen']['emb_method'])
        results_df_t['log_name'] = self.parms['gl']['file']
        results_df_t['times_included'] = include_times
        results_df_t.to_csv(os.path.join('output_files', cm.EmbeddingMethods.get_file_path(
            self.parms['t_gen']['emb_method'],
            self.parms['t_gen']['emb_method'],
            self.parms['t_gen']['include_times'],
            self.parms['t_gen']['concat_method'])), index=False)

    @staticmethod
    def _define_model_path(parms):
        inter = parms['model_type'] in ['inter', 'dual_inter', 'inter_nt']
        is_dual = parms['model_type'] == 'dual_inter'
        next_ac = parms['model_type'] == 'inter_nt'
        arpool = parms['all_r_pool']
        if inter:
            if is_dual:
                if arpool:
                    return ['_dpiapr', '_dwiapr', '_diapr']
                else:
                    return ['_dpispr', '_dwispr', '_dispr']
            else:
                if next_ac:
                    if arpool:
                        return ['_inapr']
                    else:
                        return ['_inspr']
                else:
                    if arpool:
                        return ['_iapr']
                    else:
                        return ['_ispr']
        else:
            return ['.h5', '_scaler.pkl', '_meta.json']

    @staticmethod
    def _save_times(times, parms):
        times = [{**{'output': parms['output']}, **times}]
        log_file = os.path.join('output_files', 'execution_times.csv')
        if not os.path.exists(log_file):
            open(log_file, 'w').close()
        if os.path.getsize(log_file) > 0:
            sup.create_csv_file(times, log_file, mode='a')
        else:
            sup.create_csv_file_header(times, log_file)

    # =============================================================================
    # Support methods
    # =============================================================================
    def _split_timeline(self, size: float, one_ts: bool) -> None:
        """
        Split an event log dataframe by time to peform split-validation.
        prefered method time splitting removing incomplete traces.
        If the testing set is smaller than the 10% of the log size
        the second method is sort by traces start and split taking the whole
        traces no matter if they are contained in the timeframe or not

        Parameters
        ----------
        size : float, validation percentage.
        one_ts : bool, Support only one timestamp.
        """
        key = 'end_timestamp' if one_ts else 'start_timestamp'
        # Split log data
        train, test = cm.split_log(self.log, one_ts, size)
        self.log_test = (test.sort_values(key, ascending=True).reset_index(drop=True))
        print('Number of instances in test log: {}'.format(len(self.log_test['caseid'].drop_duplicates())))
        self.log_train = copy.deepcopy(self.log)
        self.log_train.set_data(train.sort_values(key, ascending=True).reset_index(drop=True).to_dict('records'))
        print('Number of instances in train log: {}'.format(
            len(train.sort_values(key, ascending=True).reset_index(drop=True)['caseid'].drop_duplicates())))

    @staticmethod
    def _get_traces(data, one_timestamp):
        """
        returns the data splitted by caseid and ordered by start_timestamp
        """
        cases = list(set([x['caseid'] for x in data]))
        traces = list()
        for case in cases:
            order_key = 'end_timestamp' if one_timestamp else 'start_timestamp'
            trace = sorted(list(filter(lambda x: (x['caseid'] == case), data)), key=itemgetter(order_key))
            traces.append(trace)
        return traces
