
# Main model for setting up a tensorflow model.
import tensorflow as tf
import numpy as np

# Add directory to models.
import sys, os, fnmatch
from tools import path_to_dl
sys.path.append( path_to_dl() )

from model_layers.cell_models     import SpotCell
from model_layers.seg_loss        import segmentation_loss_series
from model_layers.compute_metrics import compute_metrics

from utils.tf_utils import optimizer_fun
from utils import load_data
from utils.get_data_series import round_up_series
from utils.image_functions import save_series, get_tfpn

# Empty structure object.
class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# generator for setting up different models.
class generator:

    def __init__(self, params):

        # Add all params to object.
        for p in params:

            setattr(self, p, params[p])

        # Name for this network.
        self.network_name = 'rnn_loop_bp_' + repr(self.backprop_length) + '_comb' + str(self.combiner_conv_size) + '_k' + str(self.conv_size) + '_path_' + self.arch_path

        self.network_dir = self.work_dir + '/' + self.network_name + '/'

        self.DATA = struct()

        # initialize a counter.
        self.val_counter = 0

        # Moving average loss?
        self.loss_total  = 0

        # Deal with directories.
        if not os.path.isdir(self.network_dir):
            print('Did not find existing network directory.')
            print('Making new dir: ' + self.network_dir)
            os.mkdir(self.network_dir)
            os.mkdir(self.network_dir+'/images')

        # csv file name.
        self.csv_file = self.network_dir + '/metrics_' + self.network_name + '.csv'

    def load_data(self):

        # Figure out what type of data.
        x_train, x_val, y_train, y_val, w_train, w_val, c_train, c_val = load_data.SEG_RNN(self.work_dir)

        self.DATA.x_train = x_train
        self.DATA.x_val   = x_val
        self.DATA.y_train = y_train
        self.DATA.y_val   = y_val
        self.DATA.w_train = w_train
        self.DATA.w_val   = w_val
        self.DATA.c_train = c_train
        self.DATA.c_val   = c_val

        self.n_val_batches = len(x_val)
        self.n_train_batches = len(x_train)

        # Find image size. Should be uniform across series.
        self.img_shape = x_val[0].shape

    # recurrent conv net
    def r_cnn(self):

        batch_x_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, self.backprop_length, None, None, 1])
        batch_y_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, self.backprop_length, None, None, 1])
        batch_w_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, self.backprop_length, None, None, 1])

        # Unpack input data into series
        input_series = tf.unstack(batch_x_placeholder, axis=1)

        # Initial state
        init_state = tf.placeholder(tf.float32, shape=[1, None, None, 64])

        with tf.variable_scope('spot_RCNN') as scope:
            # Initialize our cell object.
            rcnn = SpotCell(0, padding='same',
                            layer_depth=self.layer_depth,
                            conv_size=self.conv_size,
                            strides=1,
                            n_classes=self.nb_classes,
                            combiner_conv_size=self.combiner_conv_size
                            )

        # Loop over data. Options to do just forward or forward+backward.
        if self.arch_path is 'forward':

            current_state = init_state

            logit_series = []

            # Forward loop.
            for current_input in input_series:
                outputs, current_state = rcnn(current_input, current_state)

                # Logit-series.
                logit_series.append(outputs)

        elif self.arch_path is 'forward_backward':

            current_state = init_state

            # Forward loop.
            for current_input in input_series:
                outputs, current_state = rcnn(current_input, current_state)

            # Backward loop. Reverse input list, skip first element.
            input_series.reverse()
            # Logit-series.
            logit_series = []
            logit_series.append(outputs)

            for current_input in input_series[1:]:
                outputs, current_state = rcnn(current_input, current_state)
                logit_series.append(outputs)

            # Reverse logit_series.
            logit_series.reverse()

        # Calculate loss. Concatenate across series to perform all at once.
        pred_mask_series, final_loss = segmentation_loss_series(logit_series, batch_y_placeholder,
                                                                weight_series=batch_w_placeholder, mode='BCE')

        train_op = optimizer_fun( self.optimizer, final_loss, learning_rate=self.learning_rate )

        # Metrics are pixel accuracy, mean IU, mean accuracy, root_mean_squared_error
        metrics, metrics_op = compute_metrics(pred_mask_series, batch_y_placeholder, weights=batch_w_placeholder)


        self.batch_x_placeholder = batch_x_placeholder
        self.batch_y_placeholder = batch_y_placeholder
        self.batch_w_placeholder = batch_w_placeholder
        self.init_state = init_state

        self.current_state = current_state
        self.logit_series = logit_series
        self.pred_mask_series = pred_mask_series
        self.final_loss = final_loss
        self.train_op = train_op
        self.metrics = metrics
        self.metrics_op = metrics_op

    def gen_session(self):

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        # Restore previous session?
        if self.restore_sess:
            # Find last checkpoint if no name provided.
            if self.restore_name == None:
                chp_files = fnmatch.filter(os.listdir(self.network_dir), '*.ckpt.index')
                chp_files.sort()
                self.restore_name = chp_files[-1][0:-11]
            start_epoch = int(self.restore_name.split('_')[-1]) + 1
            self.saver = tf.train.import_meta_graph(self.network_dir + self.restore_name + '.ckpt.meta')
            # Restore the model from the trained network
            self.saver.restore(self.sess, self.network_dir + self.restore_name + '.ckpt')

        else:
            start_epoch = 0

            # Delete any previous checkpoint files.
            chp_files = fnmatch.filter(os.listdir(self.network_dir), '*.ckpt.*')
            for f in chp_files:
                os.remove(self.network_dir+f)

        self.epoch_range = range(start_epoch, start_epoch+self.epochs)

    def save(self):

        self.saver.save(self.sess, self.network_dir + '/RNN_SEG_' + repr(self.iteration) + '.ckpt')

    def r_cnn_train(self, idx):

        # Frames.
        img_shape = self.DATA.x_train[idx].shape
        frames = img_shape[0]

        # Get data rounded up to backprop_length. Don't need cell masks here technically...
        x, y, w, _ = self.get_round_series(idx, 'train')

        # update initial state for each call to this function. Ensures proper state size across all batches.
        curr_state = np.zeros([1, img_shape[1], img_shape[2], 64])

        for j in range(int(frames / self.backprop_length)):
            # frame-series.
            frame_idx = range(j * self.backprop_length, (j + 1) * self.backprop_length)

            # image data.
            batch_data = np.reshape(x[frame_idx], [1, self.backprop_length, img_shape[1], img_shape[2], 1])

            # Label
            batch_label = np.reshape(y[frame_idx], [1, self.backprop_length, img_shape[1], img_shape[2], 1])

            # Weights.
            batch_weight = np.reshape(w[frame_idx], [1, self.backprop_length, img_shape[1], img_shape[2], 1])

            # perform learning while fetching the state.
            self.sess.run(self.train_op, feed_dict={self.batch_x_placeholder: batch_data, self.batch_y_placeholder: batch_label, self.batch_w_placeholder: batch_weight, self.init_state: curr_state})

            curr_state = self.sess.run(self.current_state, feed_dict={self.batch_x_placeholder: batch_data, self.batch_y_placeholder: batch_label, self.batch_w_placeholder: batch_weight, self.init_state: curr_state})

    def r_cnn_val(self, idx):

        # Original image data shape.
        img_shape = self.DATA.x_val[idx].shape

        # Get data rounded up to backprop_length.
        x, y, w, c = self.get_round_series(idx, 'val')

        frames = x.shape[0]

        # Return a np array of masks for downstream stats
        r_masks = np.zeros((frames, img_shape[1], img_shape[2]))

        # update initial state for each call to this function. Ensures proper state size across all batches.
        curr_state = np.zeros([1, img_shape[1], img_shape[2], 64])

        for j in range(int(frames / self.backprop_length)):

            # frame-series.
            frame_idx = range(j * self.backprop_length, (j + 1) * self.backprop_length)

            # image data.
            batch_data = np.reshape(x[frame_idx], [1, self.backprop_length, img_shape[1], img_shape[2], 1])

            # Label
            batch_label = np.reshape(y[frame_idx], [1, self.backprop_length, img_shape[1], img_shape[2], 1])

            # Weights.
            batch_weight = np.reshape(w[frame_idx], [1, self.backprop_length, img_shape[1], img_shape[2], 1])

            # Get the state and mask series
            self.sess.run(self.metrics_op, feed_dict={self.batch_x_placeholder: batch_data, self.batch_y_placeholder: batch_label, self.batch_w_placeholder: batch_weight, self.init_state: curr_state})

            curr_state, masks, loss_temp = self.sess.run([self.current_state, self.pred_mask_series, self.final_loss],
                                                         feed_dict={self.batch_x_placeholder: batch_data, self.batch_y_placeholder: batch_label,
                                                                    self.batch_w_placeholder: batch_weight, self.init_state: curr_state})

            # condensed mask numpy array. Multiply by cell mask, c.
            r_masks[frame_idx] = np.asarray([m[0] for m in masks])
            r_masks[frame_idx] = r_masks[frame_idx] * c[frame_idx]

            #
            # Get moving average of metrics and losses
            if self.val_counter == 0:
                self.loss_total = loss_temp
            else:
                self.loss_total = (1 - 1 / (self.val_counter + 1)) * self.loss_total + 1 / (self.val_counter + 1) * loss_temp

            self.val_counter = self.val_counter + 1

        # Need to cut off arrays so that only real frames get passed onwards.
        real_frames = range(img_shape[0])

        # Return relevant data for downstream processing.
        return x[real_frames], y[real_frames], r_masks[real_frames]

    # Save labeled data and predicted seg mask.
    def save_imgs(self, x, y, masks, end_name):
        # Masks need to be list?
        out_masks = list(masks)
        out_name = self.network_dir + 'images/' + 'iteration_' + repr(self.iteration) + end_name
        save_series(x, out_masks, out_name, y)

    def get_all_metrics(self):

        out_metrics = self.sess.run(self.metrics)

        return out_metrics

    def get_round_series(self, idx, data_src):

        if data_src is "train":

            x = round_up_series(self.DATA.x_train[idx], self.backprop_length)
            y = round_up_series(self.DATA.y_train[idx], self.backprop_length)
            w = round_up_series(self.DATA.w_train[idx], self.backprop_length)
            c = round_up_series(self.DATA.c_train[idx], self.backprop_length)

        elif data_src is "val":

            x = round_up_series(self.DATA.x_val[idx], self.backprop_length)
            y = round_up_series(self.DATA.y_val[idx], self.backprop_length)
            w = round_up_series(self.DATA.w_val[idx], self.backprop_length)
            c = round_up_series(self.DATA.c_val[idx], self.backprop_length)

        return x, y, w, c

    def local_variables_initializer(self):

        self.sess.run(tf.local_variables_initializer())
