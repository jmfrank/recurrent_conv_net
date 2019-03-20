# Tools for saving images in useful formats for viewing network results.

import math
import numpy as np
import sys
from PIL import Image, ImageDraw
import skimage as sk

def save_series(x, masks, file, labels):

    # Check image size and mask size
    frames = x.shape[0]
    height = x.shape[1]
    width  = x.shape[2]

    if len(masks) != frames or masks[0].shape[0] != height or masks[0].shape[1] != width:
        print(len(masks))
        print(masks[0].shape)
        print(x.shape)
        sys.exit('image data / mask size mistmatch.')

    # Reshape data.
    x = np.moveaxis(x, 0, 2)
    masks = np.asarray(masks)
    masks = np.moveaxis(masks, 0, 2)
    labels = np.moveaxis(labels,0,2)

    # Individual Outputs will need to have a reasonable width. Keep it at 5 frames.
    n_output = int(math.ceil(frames/5))

    for i in range(n_output):

        if i == n_output -1:
            frame_range = range(i*5,frames)
        else:
            frame_range = range(i*5,(i+1)*5)

        full_width = width*len(frame_range)

        x_sub = np.reshape(x[:,:,frame_range],[height,full_width],order='F')
        masks_sub = np.reshape(masks[:,:,frame_range],[height,full_width],order='F')
        labels_sub = np.reshape(labels[:,:,frame_range],[height,full_width],order='F')

        x_sub = rescale_data(x_sub)
        masks_sub = rescale_data(masks_sub)

        output = np.concatenate((x_sub,masks_sub),0)
        img = Image.fromarray(output).convert('RGB')

        draw = ImageDraw.Draw(img,'RGBA')

        # Get centers of label regions.
        label = sk.measure.label(labels_sub)
        regions = sk.measure.regionprops(label)

        # Loop over centroids, draw an ellipse (need to shift y-location)
        for r in regions:
            C = r.centroid
            xy = (C[1]-5, C[0]-5, C[1]+5, C[0]+5)
            draw.ellipse(xy, outline=(255,0,0,255))

        img.save(file + '_series_' + format(i, '03') + '.png')


def rescale_data(x, out_type='8bit'):

    if out_type == '8bit':
        max_val = 255

    # Normalize.
    x = (x - x.min())/ (x.max()-x.min())

    # scale.
    x = (x * max_val).astype(np.uint8)

    return x


# Generate true/false positive/negative. min_size is in pixels.
# border removes objects around border of 'border_size' pixels.
def get_tfpn(y, masks):

    img_shape = (y.shape[1], y.shape[2])
    frames = y.shape[0]

    # dictionary of counts.
    counts = {"true_positive":0, "true_negative": 0, "false_positive":0, "N":0 }
    counts_list = []

    for f in range(frames):

        # ground truth.
        truth_regions = array_2_regions(y[f])
        # prediction
        pred_regions = array_2_regions(masks[f])

        these_counts = count_tfpn(truth_regions, pred_regions)
        counts_list.append(these_counts)

        counts = add_dict(counts,these_counts)

    return counts, counts_list


# add dictionaries together.
def add_dict(A,B):

    for key, value in A.items():

        A[key] = value + B[key]

    return A

# Get counts for one image.
def count_tfpn(truth_regions, pred_regions):
    # Loop over regions

    true_positive = []
    true_negative = []

    for i, r in enumerate(truth_regions):

        home = False

        # check if overlap to predicted regions.
        for j, m in enumerate(pred_regions):

            dbl_count = 0

            # check if any overlap.
            if (overlap(r.coords, m.coords)):

                true_positive.append(j)

                home = True

                dbl_count = dbl_count + 1

            if (dbl_count > 1):
                print('Detected double counts. Fix me')

        if not home:
            true_negative.append(i)

    false_positive = list(set(list(range(0, len(pred_regions)))) - set(true_positive))
    N = len(truth_regions)
    counts = {"true_positive": len(true_positive), "true_negative": len(true_negative), "false_positive": len(false_positive), "N"  : N}

    return counts


# get region props from binary mask (np array)
def array_2_regions(mask):

    label = sk.measure.label(mask)
    regions = sk.measure.regionprops(label)

    return regions


#Filter a stack of masks (first index is frames...)
def filter_mask_series(masks, min_size):

    out_shape = masks.shape
    frames = out_shape[0]

    out_series = np.zeros(out_shape)

    for f in range(frames):

        out_series[f] = filter_mask(masks[f], min_size)

    return out_series


# remove objects that are too small.
def filter_mask(mask, min_size):
    img_shape = mask.shape

    regions = array_2_regions(mask)

    out_mask = np.zeros(img_shape)

    for r in regions:

        # skip this region if too small.
        if r.coords.shape[0] < min_size:
            continue

        # Fill in mask.
        for c in r.coords:
            out_mask[c[0], c[1]] = mask[c[0], c[1]]

    return out_mask


# computers overlap of two masks. A and B are coordinats from regionprops
def overlap(A, B):
    out = False

    for a in A:

        for b in B:

            if a[0] == b[0] and a[1] == b[1]:
                out = True

    return out

