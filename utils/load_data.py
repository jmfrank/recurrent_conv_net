import pickle

def SEG_RNN(work_dir):

    f = open(work_dir+'/data.pckl','rb')
    x_train = pickle.load(f)
    x_val = pickle.load(f)
    y_train = pickle.load(f)
    y_val = pickle.load(f)
    w_train = pickle.load(f)
    w_val = pickle.load(f)
    c_train = pickle.load(f)
    c_val = pickle.load(f)

    f.close()
    
    return (x_train, x_val, y_train, y_val, w_train, w_val, c_train, c_val)



def RPN(work_dir):

    # Get the data.
    #f = open('/floyd/input/nuc_seg_synthetic/data.pckl', 'rb')
    #f = open('/floyd/input/nuc_seg/data.pckl', 'rb')
    
    f = open(work_dir+'/data.pckl','rb')
    x_train = pickle.load(f)
    x_val = pickle.load(f)
    y_train = pickle.load(f)
    y_val = pickle.load(f)
    bbox_train = pickle.load(f)
    bbox_val = pickle.load(f)
    f.close()
    
    return (x_train, x_val, bbox_train, bbox_val)

def SEG(work_dir):

    # Get the data.
    #f = open('/floyd/input/nuc_seg_synthetic/data.pckl', 'rb')
    #f = open('/floyd/input/nuc_seg/data.pckl', 'rb')
    
    f = open(work_dir+'/data.pckl','rb')
    x_train = pickle.load(f)
    x_val = pickle.load(f)
    y_train = pickle.load(f)
    y_val = pickle.load(f)
    w_train = pickle.load(f)
    w_val = pickle.load(f)
    bbox_train = pickle.load(f)
    bbox_val = pickle.load(f)
    f.close()
    
    return (x_train, x_val, y_train, y_val, w_train, w_val, bbox_train, bbox_val)



### Linfeng edits(08-23-18) ###

def load_data_full():
    """Retrieve the nucleus dataset and process the data."""
    # Set defaults.
    nb_classes = 2 #dataset dependent 
    batch_size = 1
    epochs     = 400
    
    # Input image dimensions
    #img_rows, img_cols = 64, 64
    img_rows, img_cols = 448, 448

    # Get the data.
    #f = open('/floyd/input/nuc_seg/data.pckl', 'rb')
    #f = open('/floyd/input/nuc_seg_large/data_large.pckl', 'rb')
    f = open('data_large.pckl', 'rb')
    x_train = pickle.load(f)
    x_test = pickle.load(f)
    y_train = pickle.load(f)
    y_test = pickle.load(f)
    e_train = pickle.load(f)
    e_test = pickle.load(f)
    bbox_train = pickle.load(f)
    bbox_test = pickle.load(f)
    f.close()
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    input_shape = (img_rows, img_cols, 1)
    mean_pixel = x_train.mean(axis=(0, 1, 2), keepdims=True) 
    std_pixel = x_train.std(axis=(0, 1, 2), keepdims=True) 
    x_train = (x_train - mean_pixel) / std_pixel
    x_test = (x_test - mean_pixel) / std_pixel
    # normalize the train/test labels to only contain 1 and 0
    y_train /= 255
    y_test /= 255
    
    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, e_train, e_test, bbox_train, bbox_test, epochs)

### Linfeng edits end ###
