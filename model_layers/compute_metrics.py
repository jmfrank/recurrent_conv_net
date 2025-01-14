import tensorflow as tf

def compute_metrics(pred_masks, labels, batch_size = 1, weights=None):
    
    all_metrics = {}
    all_metrics_op = {}
    
    all_metrics['global'] = {}

    all_metrics_op['global'] = {}

    
    # format the argument for the metrics
    labels_reshape = tf.cast(tf.reshape(labels,[batch_size, -1]), tf.int32)
    pred_masks_reshape = tf.cast(tf.reshape(pred_masks,[batch_size, -1]), tf.int32)

    if weights is not None:
        weights_reshape   = tf.cast(tf.reshape(weights,[batch_size,-1]),tf.int32)
    else:
        weights_reshape = None

    # Global metrics
    all_metrics['global']['pixel_accuracy'], all_metrics_op['global']['pixel_accuracy'] = tf.metrics.accuracy(labels_reshape, pred_masks_reshape, weights = weights_reshape)
    
    all_metrics['global']['mean_IU'], all_metrics_op['global']['mean_IU'] = tf.metrics.mean_iou(labels_reshape, pred_masks_reshape, num_classes = 2, weights = weights_reshape)
    
    all_metrics['global']['rmse'], all_metrics_op['global']['rmse'] = tf.metrics.root_mean_squared_error(labels_reshape, pred_masks_reshape, weights = weights_reshape)
    
    all_metrics['global']['precision'], all_metrics_op['global']['precision'] = tf.metrics.precision(labels_reshape, pred_masks_reshape, weights=weights_reshape)
    
    all_metrics['global']['recall'], all_metrics_op['global']['recall'] = tf.metrics.recall(labels_reshape, pred_masks_reshape, weights=weights_reshape)


    
    return all_metrics, all_metrics_op
