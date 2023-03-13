# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""
import multiprocessing
import os
import sys

import click
import yaml

import deep_simulator as ds


@click.command()
@click.option('--file', default=None, required=True, type=str)
@click.option('--update_gen/--no-update_gen', default=False, required=False, type=bool)
@click.option('--update_ia_gen/--no-update_ia_gen', default=False, required=False, type=bool)
@click.option('--update_mpdf_gen/--no-update_mpdf_gen', default=False, required=False, type=bool)
@click.option('--update_times_gen/--no-update_times_gen', default=False, required=False, type=bool)
@click.option('--save_models/--no-save_models', default=True, required=False, type=bool)
@click.option('--evaluate/--no-evaluate', default=True, required=False, type=bool)
@click.option('--mining_alg', default='sm1', required=False, type=click.Choice(['sm1', 'sm2', 'sm3']))
@click.option('--s_gen_repetitions', default=5, required=False, type=int)
@click.option('--s_gen_max_eval', default=30, required=False, type=int)
@click.option('--t_gen_epochs', default=100, required=False, type=int)
@click.option('--t_gen_max_eval', default=6, required=False, type=int)
@click.option('--emb_method', default="emb_dot_product", required=False, type=str)
@click.option('--concat_method', default="single_sentence", required=False, type=str)
@click.option('--include_times', default=False, required=False, type=bool)

def main(file, update_gen, update_ia_gen, update_mpdf_gen, update_times_gen, save_models, evaluate, mining_alg,
         s_gen_repetitions, s_gen_max_eval, t_gen_epochs, t_gen_max_eval, emb_method, concat_method, include_times):
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
    params['i_gen']['epochs'] = 100
    params['i_gen']['gen_method'] = 'prophet'  # pdf, dl, mul_pdf, test, prophet
    # Times allocator parameters
    params['t_gen'] = dict()
    params['t_gen']['emb_method'] = emb_method # emb_dot_product, emb_dot_product_times, emb_dot_product_act_weighting, emb_w2vec
    params['t_gen']['concat_method'] = concat_method # single_sentence, full_sentence, weighting //Solo aplica si se escoge w2vec como embedding method
    params['t_gen']['include_times'] = include_times # True, False // Solo aplica si se escoge w2vec o dot product con weighting
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
    params['t_gen']['opt_method'] = 'bayesian'  # bayesian, rand_hpc
    params['t_gen']['all_r_pool'] = True  # only intercase features
    params['t_gen']['reschedule'] = False  # reschedule according resource pool ocupation
    params['t_gen']['rp_similarity'] = 0.80  # Train models

    simulator = ds.DeepSimulator(params)
    simulator.execute_pipeline()


def read_properties(params):
    """ Sets the app general parms"""
    properties_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'properties.yml')
    with open(properties_path, 'r') as f:
        properties = yaml.load(f, Loader=yaml.FullLoader)
    if properties is None:
        raise ValueError('Properties is empty')
    params['gl'] = {**params['gl'], **properties}
    return params


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])

