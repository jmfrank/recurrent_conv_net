import tensorflow as tf


#Function for performing segmentation loss over a series.
def segmentation_loss_series( logit_series, label_series, weight_series=None, mode = 'BCE', nb_classes = 2, contour = None):

    with tf.variable_scope("loss"):
        final_loss = tf.Variable(0, dtype=tf.float32)
    
    pred_mask_series = [] 
    
    for i in range(len(logit_series)):

        logits = logit_series[i]
        pred_mask = tf.argmax(logits, dimension=3)
        # pred_masks = tf.reshape(logits,[input_height,input_width])
        pred_mask = tf.to_float(pred_mask)
        pred_mask_series.append(pred_mask)
        
        labels = label_series[:,i]

        if weight_series is not None:
            weights = weight_series[:,i]
        else:
            weights = None

        seg_loss = segmentation_loss(logits, labels, weights=weights, mode=mode, nb_classes = nb_classes, contour = contour)

        final_loss = final_loss + seg_loss

    return pred_mask_series, final_loss


def segmentation_loss(final_logits, labels, weights = None, mode = 'BCE', nb_classes = 2, contour=None):
    
    """
    if use_dice:
        # based on the watershed lines, tune the unet featmaps have 1 at the background predictions
        contour = tf.reshape(contour, labels.shape)

        # contour_reverse only has 1 at the watershed lines
        contour_reverse = 1 - contour
        contour_reverse_float = tf.to_float(contour_reverse)

        contour_combined_mask = tf.concat(values=[contour_reverse_float, contour_reverse_float - 1], axis=3)
        final_logits = tf.maximum(final_logits, contour_combined_mask)
    """
    
    # Loss as pixel-wise cross-entropy according to 'Fully convolutional networks for semantic segmentation'
    # shape of Y: batch_size, height, width, num_classes
    flat_logits = tf.reshape(final_logits, shape=(-1,nb_classes))    
    input_shape = tf.shape(labels)
    # Convert y_train to be able to use pixel-wise cross entropy
    train_class_label = tf.equal(labels, 1)
    train_background_label = tf.not_equal(labels, 1)
    # Convert the boolean values into floats 
    train_mask_class = tf.to_float(train_class_label)
    train_mask_background = tf.to_float(train_background_label)
    
    train_combined_mask = tf.concat(values=[train_mask_background, train_mask_class], axis=3)
    
    # Reshape to fit to tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
    flat_labels = tf.reshape(train_combined_mask, shape=(-1, nb_classes))
    # Reshape edge matrix to mutiply with the entropy
    
    if mode == 'DICE':
        seg_loss = soft_dice_loss(flat_labels, flat_logits, epsilon=1e-6)
    elif mode == 'BCE':
        seg_loss = cross_entropie_loss(flat_labels, flat_logits, input_shape, weights = weights)
    elif mode == 'COMBO':
        seg_loss = cross_entropie_loss(flat_labels, flat_logits, input_shape, weights = weights) + soft_dice_loss(flat_labels, flat_logits, epsilon=1e-6)
    
    return seg_loss
    
    
def cross_entropie_loss(y_true, y_pred, input_shape, weights = None ):
    
    cross_entropies = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=y_true)
    
    # Reshape the cross_entropies to match the size of edge_weights
    cross_entropies_reshape = tf.reshape(cross_entropies,input_shape)
    
    # Weight if needed. 
    if weights is not None:
        # Apply the edge matrix to the entropy for the final loss function
        cross_entropies = tf.multiply(cross_entropies_reshape, weights)

    cross_entropy_sum = tf.reduce_sum(cross_entropies)
    
    return cross_entropy_sum


def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    ''' 
    This is an implementation created by @jeremyjordan
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    
    # skip the batch and class axis for calculating Dice score
    # axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * tf.reduce_sum(y_pred * y_true)
    denominator = tf.reduce_sum(tf.square(y_pred) + tf.square(y_true))
    
    return 1 - tf.reduce_mean(numerator / (denominator + epsilon)) # average over classes and batch

