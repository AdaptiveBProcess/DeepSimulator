# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 13:27:58 2020

@author: Manuel Camargo
"""
import os
import sys
import getopt
import multiprocessing

import deep_simulator as ds

def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file', '-g': 'update_gen', 
              '-i': 'update_ia_gen', '-p': 'update_mpdf_gen', 
              '-t': 'update_times_gen', '-s': 'save_models', 
              '-e': 'evaluate', '-m': 'mining_alg'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)

# --setup--
def main(argv):
    """Main aplication method"""
    parms = dict()
    parms = define_general_parms(parms)
    # parms setting manual fixed or catched by console
    if not argv:
        # Event-log parms
        parms['gl']['file'] = 'PurchasingExample.xes'
        parms['gl']['update_gen'] = False
        parms['gl']['update_ia_gen'] = False
        parms['gl']['update_mpdf_gen'] = False
        parms['gl']['update_times_gen'] = False
        parms['gl']['save_models'] = True
        parms['gl']['evaluate'] = True
        parms['gl']['mining_alg'] = 'sm3'
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt( argv, "h:f:g:i:p:t:s:e:m:",
                ['file=','update_gen=', 'update_ia_gen=', 'update_mpdf_gen=', 
                 'update_times_gen=', 'save_models=', 'evaluate=', 
                 'mining_alg='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parms['gl'][key] = None
                elif key in ['update_gen', 'update_ia_gen', 'update_mpdf_gen',
                             'update_times_gen', 'save_models', 'evaluate']:
                    parms['gl'][key] = arg in ['True', 'true', 1]
                else:
                    parms['gl'][key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)
    parms['gl']['sim_metric'] = 'tsd' # Main metric
    # Additional metrics
    parms['gl']['add_metrics'] = ['day_hour_emd', 'log_mae', 'dl', 'mae']
    parms['gl']['exp_reps'] = 1
    # Sequences generator
    parms['s_gen'] = dict()
    parms['s_gen']['repetitions'] = 5
    parms['s_gen']['max_eval'] = 15
    parms['s_gen']['concurrency'] = [0.0, 1.0]
    parms['s_gen']['epsilon'] = [0.0, 1.0]
    parms['s_gen']['eta'] = [0.0, 1.0]
    parms['s_gen']['alg_manag'] = ['replacement', 'repair']
    parms['s_gen']['gate_management'] = ['discovery', 'equiprobable']
    # Inter arrival generator
    parms['i_gen'] = dict()
    parms['i_gen']['batch_size'] = 32 # Usually 32/64/128/256
    parms['i_gen']['epochs'] = 200
    parms['i_gen']['gen_method'] = 'dl' # pdf, dl, mul_pdf
    # Times allocator parameters
    parms['t_gen'] = dict()
    parms['t_gen']['imp'] = 1
    parms['t_gen']['max_eval'] = 12
    parms['t_gen']['batch_size'] = 32 # Usually 32/64/128/256
    parms['t_gen']['epochs'] = 200
    parms['t_gen']['n_size'] = [5, 10, 15]
    parms['t_gen']['l_size'] = [50, 100] 
    parms['t_gen']['lstm_act'] = ['selu', 'tanh']
    parms['t_gen']['dense_act'] = ['linear']
    parms['t_gen']['optim'] = ['Nadam']
    parms['t_gen']['model_type'] = 'basic'
    parms['t_gen']['opt_method'] = 'rand_hpc'
    # Train models
    print(parms)
    simulator = ds.DeepSimulator(parms)
    simulator.execute_pipeline()

def define_general_parms(parms):
    """ Sets the app general parms"""
    column_names = {'Case ID': 'caseid',
                    'Activity': 'task',
                    'lifecycle:transition': 'event_type',
                    'Resource': 'user'}
    parms['gl'] = dict()
    # Event-log reading options
    parms['gl']['read_options'] = {
            'timeformat': '%Y-%m-%dT%H:%M:%S.%f',
            'column_names': column_names,
            'one_timestamp': False,
            'filter_d_attrib': True}
    # Folders structure
    parms['gl']['event_logs_path'] = os.path.join('input_files', 'event_logs')
    parms['gl']['bpmn_models'] = os.path.join('input_files', 'bpmn_models')
    parms['gl']['embedded_path'] = os.path.join('input_files', 'embedded_matix')
    parms['gl']['ia_gen_path'] = os.path.join('input_files', 'ia_gen_models')
    parms['gl']['seq_flow_gen_path'] = os.path.join('input_files', 'seq_flow_gen_models')
    parms['gl']['times_gen_path'] = os.path.join('input_files', 'times_gen_models')
    # External tools routes
    parms['gl']['sm2_path'] = os.path.join('external_tools',
                                     'splitminer2',
                                     'sm2.jar')
    parms['gl']['sm1_path'] = os.path.join('external_tools',
                                     'splitminer',
                                     'splitminer.jar')
    parms['gl']['sm3_path'] = os.path.join('external_tools',
                                     'splitminer3',
                                     'bpmtk.jar')
    parms['gl']['bimp_path'] = os.path.join('external_tools',
                                      'bimp',
                                      'qbp-simulator-engine.jar')
    parms['gl']['align_path'] = os.path.join('external_tools',
                                       'proconformance',
                                       'ProConformance2.jar')
    parms['gl']['calender_path'] = os.path.join('external_tools',
                                          'calenderimp',
                                          'CalenderImp.jar')
    parms['gl']['simulator'] = 'bimp'
    return parms


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])
