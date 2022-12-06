# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 10:02:19 2021

@author: Manuel Camargo
"""
import os
import sys
import getopt
import shutil

import utils.support as sup

from what_if_modules import embedding_updater as eu
from what_if_modules import times_predictor_updater as tpu


class ModelsUpdater:
    """
    This class evaluates the inter-arrival times
    """

    def __init__(self, parms):
        self.parms = parms
        self.modified_name = self.parms['gl']['modified_file'].split('.')[0]
        self.complete_name = self.parms['gl']['complete_file'].split('.')[0]

    def update_models(self):
        print('Updating embedding...')
        emb_updater = eu.EmbeddingUpdater(self.parms)
        emb_updater.execute_pipeline()
        print('Updating predictive models...')
        model_updater = tpu.TimesPredictorUpdater(
            emb_updater.new_ac_weights,
            emb_updater.modif_params,
            os.path.join(self.parms['gl']['times_gen_path'],
                         self.modified_name))
        model_updater.execute_pipeline()

    def save_models(self):
        output_path = os.path.join('output_files', sup.folder_id())
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            for folder in ['bpmn_models', 'embedded_matix', 'event_logs',
                           'ia_gen_models', 'times_gen_models']:
                os.makedirs(os.path.join(output_path, folder))
                dest_dir = os.path.join(output_path, folder)
                if folder == 'bpmn_models':
                    src_dir = self.parms['gl']['bpmn_models']
                    shutil.copy(os.path.join(src_dir, self.complete_name + '.bpmn'),
                                os.path.join(dest_dir, self.modified_name + '_upd.bpmn'))
                    shutil.copy(os.path.join(src_dir, self.complete_name + '_meta.json'),
                                os.path.join(dest_dir, self.modified_name + '_upd_meta.json'))
                elif folder == 'embedded_matix':
                    src_dir = self.parms['gl']['embedded_path']
                    shutil.copy(os.path.join(src_dir, 'ac_' + self.modified_name + '_upd.emb'),
                                os.path.join(dest_dir, 'ac_' + self.modified_name + '_upd.emb'))
                    shutil.copy(os.path.join(src_dir, self.modified_name + '_upd_emb.h5'),
                                os.path.join(dest_dir, self.modified_name + '_upd_emb.h5'))
                elif folder == 'event_logs':
                    src_dir = self.parms['gl']['event_logs_path']
                    shutil.copy(os.path.join(src_dir,
                                             self.complete_name + '.xes'),
                                os.path.join(dest_dir,
                                             self.modified_name + '_upd.xes'))
                elif folder == 'ia_gen_models':
                    src_dir = self.parms['gl']['ia_gen_path']
                    shutil.copy(
                        os.path.join(
                            src_dir, self.complete_name + '_prf.json'),
                        os.path.join(
                            dest_dir, self.modified_name + '_upd_prf.json'))
                    shutil.copy(
                        os.path.join(
                            src_dir, self.complete_name + '_prf_meta.json'),
                        os.path.join(
                            dest_dir, self.modified_name + '_upd_prf_meta.json'))
                elif folder == 'times_gen_models':
                    src_dir = self.parms['gl']['times_gen_path']
                    for ext in ['_upd_diapr_meta.json', '_upd_dwiapr.h5',
                                '_upd_dpiapr.h5']:
                        shutil.copy(
                            os.path.join(src_dir, self.modified_name + ext),
                            os.path.join(dest_dir, self.modified_name + ext))
                    for ext in ['_diapr_end_inter_scaler.pkl',
                                '_diapr_inter_scaler.pkl', '_diapr_scaler.pkl', ]:
                        shutil.copy(
                            os.path.join(src_dir, self.modified_name + ext),
                            os.path.join(dest_dir, self.modified_name + '_upd' + ext))


# =============================================================================
# General methods
# =============================================================================

def catch_parameter(opt):
    """Change the captured parameters names"""
    switch = {'-h': 'help', '-f': 'file'}
    try:
        return switch[opt]
    except:
        raise Exception('Invalid option ' + opt)


def main(argv):
    """Main aplication method"""
    parms = dict()
    parms = define_general_parms(parms)
    # parms setting manual fixed or catched by console
    if not argv:
        # Event-log parms
        parms['gl']['modified_file'] = 'BPI_Challenge_2017_W_Two_TS_modif.xes'
        parms['gl']['complete_file'] = 'BPI_Challenge_2017_W_Two_TS.xes'
    else:
        # Catch parms by console
        try:
            opts, _ = getopt.getopt(argv, "h:f:", ['file='])
            for opt, arg in opts:
                key = catch_parameter(opt)
                if arg in ['None', 'none']:
                    parms['gl'][key] = None
                else:
                    parms['gl'][key] = arg
        except getopt.GetoptError:
            print('Invalid option')
            sys.exit(2)

    models_updater = ModelsUpdater(parms)
    models_updater.update_models()
    models_updater.save_models()


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
    parms['gl']['times_gen_path'] = os.path.join('input_files', 'times_gen_models')
    return parms


if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    main(sys.argv[1:])
