# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""
import os
import sys

import click
import yaml

import deep_simulator as ds
from support_modules.common import EmbeddingMethods as Em, OUTPUT_FILES
from support_modules.common import InterArrivalGenerativeMethods as IaG
from support_modules.common import SequencesGenerativeMethods as SqG
from support_modules.common import SplitMinerVersion as Sm
from support_modules.common import W2VecConcatMethod as Cm


@click.command()
@click.option('--file', default=None, required=True, type=str)
@click.option('--update_gen/--no-update_gen', default=False, required=False, type=bool)
@click.option('--update_ia_gen/--no-update_ia_gen', default=False, required=False, type=bool)
@click.option('--update_mpdf_gen/--no-update_mpdf_gen', default=False, required=False, type=bool)
@click.option('--update_times_gen/--no-update_times_gen', default=False, required=False, type=bool)
@click.option('--save_models/--no-save_models', default=True, required=False, type=bool)
@click.option('--evaluate/--no-evaluate', default=True, required=False, type=bool)
@click.option('--mining_alg', default=Sm.SM_V1, required=False, type=click.Choice(Sm().get_methods()))
@click.option('--s_gen_repetitions', default=5, required=False, type=int)
@click.option('--s_gen_max_eval', default=30, required=False, type=int)
@click.option('--t_gen_epochs', default=200, required=False, type=int)
@click.option('--t_gen_max_eval', default=12, required=False, type=int)
@click.option('--seq_gen_method', default=SqG.PROCESS_MODEL, required=False, type=click.Choice(SqG().get_methods()))
@click.option('--ia_gen_method', default=IaG.PROPHET, required=False, type=click.Choice(IaG().get_methods()))
@click.option('--emb_method', default=Em.DOT_PROD, required=False, type=click.Choice(Em().get_types()))
@click.option('--concat_method', default=Cm.SINGLE_SENTENCE, required=False, type=click.Choice(Cm().get_methods()))
@click.option('--include_times', default=False, required=False, type=bool)
def main(file, update_gen, update_ia_gen, update_mpdf_gen, update_times_gen, save_models, evaluate, mining_alg,
         s_gen_repetitions, s_gen_max_eval, t_gen_epochs, t_gen_max_eval, seq_gen_method, ia_gen_method, emb_method,
         concat_method, include_times):
    params = dict()
    params['gl'] = dict()
    params['gl']['file'] = file
    params['gl']['update_gen'] = update_gen
    params['gl']['update_ia_gen'] = update_ia_gen
    params['gl']['update_mpdf_gen'] = update_mpdf_gen
    params['gl']['update_times_gen'] = update_times_gen
    params['gl']['save_models'] = save_models
    params['gl']['evaluate'] = evaluate
    params['gl']['mining_alg'] = mining_alg
    params = read_properties(params)
    params['gl']['sim_metric'] = 'tsd'  # Main metric
    # Additional metrics
    params['gl']['add_metrics'] = ['day_hour_emd', 'log_mae', 'dl', 'mae']
    params['gl']['exp_reps'] = 1
    # Sequences generator
    params['s_gen'] = dict()
    params['s_gen']['gen_method'] = seq_gen_method  # stochastic_process_model, test
    params['s_gen']['repetitions'] = s_gen_repetitions
    params['s_gen']['max_eval'] = s_gen_max_eval
    params['s_gen']['concurrency'] = [0.0, 1.0]
    params['s_gen']['epsilon'] = [0.0, 1.0]
    params['s_gen']['eta'] = [0.0, 1.0]
    params['s_gen']['alg_manag'] = ['replacement', 'repair']
    params['s_gen']['gate_management'] = ['discovery', 'equiprobable']
    # Inter arrival generator
    params['i_gen'] = dict()
    params['i_gen']['batch_size'] = 32  # Usually 32/64/128/256
    params['i_gen']['epochs'] = 2
    params['i_gen']['gen_method'] = ia_gen_method  # pdf, dl, mul_pdf, test, prophet
    # Times allocator parameters
    params['t_gen'] = dict()
    # emb_dot_product, emb_dot_product_times, emb_dot_product_act_weighting, emb_w2vec
    params['t_gen']['emb_method'] = emb_method
    # single_sentence, full_sentence, weighting: Only applies if w2vec is chosen as embedding method
    params['t_gen']['concat_method'] = concat_method
    # True, False: Only applies if w2vec is chosen as embedding method with weighting
    params['t_gen']['include_times'] = include_times
    params['t_gen']['imp'] = 1
    params['t_gen']['max_eval'] = t_gen_max_eval
    params['t_gen']['batch_size'] = 32  # Usually 32/64/128/256
    params['t_gen']['epochs'] = t_gen_epochs
    params['t_gen']['n_size'] = [5, 10, 15]
    params['t_gen']['l_size'] = [50, 100]
    params['t_gen']['lstm_act'] = ['selu', 'tanh']
    params['t_gen']['dense_act'] = ['linear']
    params['t_gen']['optim'] = ['Nadam']
    params['t_gen']['model_type'] = 'dual_inter'  # basic, inter, dual_inter, inter_nt
    params['t_gen']['opt_method'] = 'rand_hpc'  # bayesian, rand_hpc
    params['t_gen']['all_r_pool'] = True  # only inter-case features
    params['t_gen']['reschedule'] = False  # reschedule according resource pool occupation
    params['t_gen']['rp_similarity'] = 0.80  # Train models

    _ensure_locations(params)

    simulator = ds.DeepSimulator(params)
    simulator.execute_pipeline()


def read_properties(params):
    """ Sets the app general params"""
    properties_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'properties.yml')
    with open(properties_path, 'r') as f:
        properties = yaml.load(f, Loader=yaml.FullLoader)
    if properties is None:
        raise ValueError('Properties is empty')
    paths = {k: os.path.join(*path.split('\\')) for k, path in properties.pop('paths').items()}
    params['gl'] = {**params['gl'], **properties, **paths}
    return params


def _ensure_locations(params):
    for folder in ['event_logs_path', 'bpmn_models', 'embedded_path', 'ia_gen_path', 'times_gen_path']:
        if not os.path.exists(params['gl'][folder]):
            os.makedirs(params['gl'][folder])
    if not os.path.exists(OUTPUT_FILES):
        os.makedirs(OUTPUT_FILES)


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])
