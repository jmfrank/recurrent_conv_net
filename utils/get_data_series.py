import numpy as np
import math


# Get data series that's a multiple of bp_length in order to ensure the recurrent network size is satisfied.
def round_up_series(x, bp_length):

    # Frames
    frames = x.shape[0]

    # Length of output (multiple of bp_length).
    N = math.ceil(frames/bp_length)*bp_length

    # get oscillating indices
    I = i_series(frames, N)

    # Output variables.
    _x = x[I]

    return _x


# Generate a list of oscillating integers used as frame index in get_data_series.
def i_series(M, N):

    dir = 1
    max_val = M - 1
    i_list = np.zeros(N,dtype=np.intp)

    if max_val > 0:

        for i in range(N-1):

            if i_list[i] + dir > max_val or i_list[i] + dir < 0:
                dir = dir*-1

            i_list[i+1] = i_list[i]+dir

    return i_list