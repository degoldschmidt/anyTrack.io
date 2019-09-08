import numpy as np

def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.array(inarray, dtype=np.int32)                  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i]) # simply return array runlengths runlen, positions and states

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate(arr):
    nans, x= nan_helper(arr)
    arr[nans]= np.interp(x(nans), x(~nans), arr[~nans])
    return arr

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def px(val):
    return int(round(val))

def dist(pos1, pos2):
    dx, dy = (pos1[0] - pos2[0]), (pos1[1] - pos2[1])
    return np.sqrt(dx * dx + dy * dy)

def in_roi(pos, roi):
    if dist(pos,(roi['x'], roi['y']))<=1.1*roi['outer']:
        return True
    else:
        return False
