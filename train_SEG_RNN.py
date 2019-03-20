# Testing the RNN network.
import os
import numpy as np

import csv

from PIL import Image
from tqdm import tqdm

# Add directory to models.
import sys
from tools import path_to_dl
sys.path.append( path_to_dl() )

from model_layers.generator import generator
from utils.image_functions import add_dict, get_tfpn

# adapted from https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/rpn_test.py
def train_and_score(params):

    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
    return:
        accuracy (float)
    """

    # generate the model.
    model = generator(params)
    model.load_data()
    model.r_cnn()

    # Get session.
    model.gen_session()

    with open(model.csv_file, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'loss', 'mean_IU',  'pixel_accuracy', 'f1', 'rmse', 'true_positive','true_negative','false_positive','N']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        idx = np.arange(model.n_train_batches)
        
        for iteration in model.epoch_range:

            model.iteration = iteration

            model.local_variables_initializer()

            # Training loop.
            np.random.shuffle(idx)

            for i in tqdm(range(0, model.n_train_batches)):

                model.r_cnn_train(idx[i])

            # Validation/testing.
            stats = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "N": 0}

            for i in range(model.n_val_batches):

                x, y, masks = model.r_cnn_val(i)

                # filter masks
                masks = filter_mask_series(masks,min_size=params['min_size'])

                # skip tfpn calculation for first iteration.
                if iteration > 0:
                    these_stats, _ = get_tfpn(y, masks, min_size=1)

                    stats = add_dict(stats, these_stats)

                if model.iteration > 0 and (model.iteration % model.checkpointN == 0 or model.iteration == model.epochs):
                    model.save_imgs(x, y, masks, '_batch_' + str(i) )


            # Global metrics.
            metrics_all = model.get_all_metrics()

            _mean_IU = metrics_all['global']['mean_IU']
            _pixel_accuracy = metrics_all['global']['pixel_accuracy']
            _f1 = 2 * _mean_IU / (1 + _mean_IU)
            _rmse = metrics_all['global']['rmse']
            write_dict = {
                'Epoch': str(model.iteration),
                'loss': str(model.loss_total),
                'mean_IU': str(_mean_IU),
                'pixel_accuracy': str(_pixel_accuracy),
                'f1': str(_f1),
                'rmse': str(_rmse),
                'true_positive': str(stats['true_positive']),
                'true_negative': str(stats['true_negative']),
                'false_positive': str(stats['false_positive']),
                'N': str(stats['N'])
            }
            writer.writerow( write_dict )
            if stats['N'] > 0:

                print('Epoch: %d - loss: %.2f - true_positive: %.2f - true_negative: %.2f' % (iteration, model.loss_total, stats['true_positive']/stats['N'], stats['true_negative']/stats['N']))
            
            if model.iteration % model.checkpointN == 0 or model.iteration == model.epochs:

                model.save()

                
    model.sess.close()


