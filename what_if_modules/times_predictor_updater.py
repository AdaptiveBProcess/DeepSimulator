# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:34:07 2021

@author: Manuel Camargo
"""

class TimesPredictorUpdater():
    """
    """

    def __init__(self, num_categories, num_users, embedding_size):
        self.num_categories = num_categories
        self.num_users = num_users
        self.embedding_size = embedding_size

    def read_predictive_model(parms, model_ext, parms_exts):
        # Load old model
        model_path = os.path.join(parms['gl']['times_gen_path'],
                                  parms['gl']['file'].split('.')[0]+model_ext+'.h5')
        model = load_model(model_path)
        model.summary()
        model_weights = {layer.name: layer.get_weights() for layer in model.layers}
        
        # Read metadata data
        data_path = os.path.join(
            parms['gl']['times_gen_path'],
            parms['gl']['file'].split('.')[0]+parms_exts+'_meta.json')
        
        with open(data_path) as file:
            parameters = json.load(file)
    
        parameters = {
            **parameters,
            **{'ac_input_shape': model.get_layer('ac_input').input_shape[0],
               'features_shape': model.get_layer('features').input_shape[0],
               'ac_emb_inp_lenght': model.get_layer('ac_embedding').input_length,
               'time_output_lenght': model.get_layer('time_output').units,
               'imp': 1}}
        return model_weights, parameters
    
    
    def create_pred_model(ac_weights, parms, names):
        # Input layer
        ac_input = Input(shape=(parms['ac_input_shape'][1], ), name='ac_input')
        features = Input(shape=(parms['features_shape'][1], 
                                parms['features_shape'][2]),
                         name='features')
    
        # Embedding layer for categorical attributes
        ac_embedding = Embedding(ac_weights.shape[0],
                                 ac_weights.shape[1],
                                 weights=[ac_weights],
                                 input_length=parms['ac_emb_inp_lenght'],
                                 trainable=False, name='ac_embedding')(ac_input)
    
        # Layer 1
        merged = Concatenate(name='concatenated', axis=2)([ac_embedding, features])
    
        l1_c1 = LSTM(parms['l_size'],
                     kernel_initializer='glorot_uniform',
                     return_sequences=True,
                     dropout=0.2,
                     implementation=parms['imp'],
                     name=names[0])(merged)
    
        # Batch Normalization Layer
        batch1 = BatchNormalization(name=names[1])(l1_c1)
    
        # The layer specialized in prediction
        l2_c1 = LSTM(parms['l_size'],
                    activation=parms['lstm_act'],
                    kernel_initializer='glorot_uniform',
                    return_sequences=False,
                    dropout=0.2,
                    implementation=parms['imp'],
                    name=names[2])(batch1)
    
        # Output Layer
        times_output = Dense(parms['time_output_lenght'],
                             activation=parms['dense_act'],
                             kernel_initializer='glorot_uniform',
                             name='time_output')(l2_c1)
    
        model = Model(inputs=[ac_input, features],
                      outputs=[times_output])
        
        if parms['optim'] == 'Nadam':
            opt = Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
        elif parms['optim'] == 'Adam':
            opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        elif parms['optim'] == 'SGD':
            opt = SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
        elif parms['optim'] == 'Adagrad':
            opt = Adagrad(learning_rate=0.01)
    
        model.compile(loss={'time_output': 'mae'}, optimizer=opt)
    
        model.summary()
        return model