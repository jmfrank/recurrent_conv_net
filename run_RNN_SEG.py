"""
Script which bypasses the evolution algorithm to test different neural network structures
"""

from train_SEG_RNN import train_and_score
import os

# params are the training hyperparameters
params = {}
params['learning_rate'] = 1e-6
params['optimizer'] = 'rmsprop' #'adam'
params['nb_classes'] = 2
params['batch_size'] = 1
params['epochs'] = 30
params['work_dir'] = os.getcwd()
params['checkpointN'] = 1
params['weight_train'] = False
params['layer_depth'] = 64
params['conv_size'] = 6
params['combiner_conv_size'] = 6
params['backprop_length'] = 6
params['arch_path'] = 'forward' # Can be forward or forward_backward

params['restore_sess'] = 1
params['restore_name'] = 'RNN_SEG_49'


train_and_score(params)
