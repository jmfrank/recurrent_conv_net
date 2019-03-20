# recurrent_spot_net

This is a recurrent convolutional neural network used for recognizing nascent transcription pulses. By using a recurrent CNN, we can detect very faint spots that are on the tail ends of a transcription burst. Because these pulses are relatively small features, typical down-sampling / upsampling approaches (i.e. U-net) cannot be applied as the feature can become lost in down-sampling. Further, the pixel-limited nature of the features mean the majority of pixels are background, and the loss function must only be taken for pixels near the features to put more weight on foreground pixels.  

The 'data' folder contains 2D time-series images of individual nuclei that show nascent transcription bursts. The data files contain the results of a optimized semantic segmentation and human-corrected results. Using utils.load_data.SEG_RNN will provide the raw images, labels, spot-masks (used as weights), and cell-masks for training and validation. (First, use the preprocess_spots.ipynb notebook to create the pickle file). 

The train_RNN_SEG.py script is a template for running training and validation. The 'generator' class controls how tensorflow builds the model. The individual recurrent CNN cell is defined in model_layers.cell_models.SpotCell. 

# Parameters 

Parameters are defined in 'run_RNN_SEG.py'. The non-obvious parameters are explained below:

The scripts will automatically create subdirectories in 'work_dir' for different model architectures.

checkpoinN: The frequency at which to save the model in terms of iterations. 


conv_size = Size of the convolution kernel for standard 2D convolution layers. 
combiner_conv_size = Size of the kernel for convolving the previous state with the current frame. This should be relatively large to account for any frame-to-frame displacement of the nascent spot. 

arch_path = Option to pass the recurrent state forward only ('forward'), or run the model forward, then backward ('forward_backward')

Number of time-points to use for each batch.
params['backprop_length'] = 6

Data-series that are less than this value are simply repeated by reversing the time-series. For example a 3-frame data series with a backprop_length=6 would be fed into the model as a 6-frame series with frame index as: [0, 1, 2, 1, 0, 1]. Is the data is longer than backprop_length, then the batch is broken up into pieces. For example a 9-frame data series with backprop_length=6 would be two batches with frame indices:
[0,1,2,3,4,5]
[6,7,8,7,6,7]

This approach maintains the time-relationship between frames and allows a constant data size for each batch rather than rebuilding the graph for each batch with variable length of time-points. The output from validation is automatically put together as the original sequence of frames. 

This does however weigh individual frames in shorter series higher because they are repeated. Other ideas to approach this are welcome! 

Choice to restore a previous session. If restore_sess=1 and retore_name=None, the latest model is taken.
params['restore_sess'] = 0
params['restore_name'] = None

# Training/validation

run_RNN_SEG will generate a sub-directory "rnn_loop_bp_X_path_Y". X is the number of frames in backprop and Y is the arch_path parameter. 

The model states are saved in this directory and if validation images are saved then the sub-dir 'images' will be filled populated with images of the validation for each iteration requested to be saved. 
