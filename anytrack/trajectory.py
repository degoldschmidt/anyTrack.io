import numpy as np
import pandas as pd

class Trajectory(object):
    def __init__(self, num_frames):
        self.columns = ['body_x', 'body_y', 'head_x', 'head_y', 'major', 'minor', 'angle', 'avg_px']
        self.all_columns = ['body_x', 'body_y', 'head_x', 'head_y', 'major', 'minor', 'angle', 'avg_px',
                            'ax', 'ay', 'ox', 'oy', 'apx', 'opx']
        self.data = np.zeros((num_frames,len(self.columns)+6))
        self.data[:] = np.nan
        self.x = self.data[:,0]
        self.y = self.data[:,1]
        self.hx = self.data[:,2]
        self.hy = self.data[:,3]
        self.major = self.data[:,4]
        self.minor = self.data[:,5]
        self.angle = self.data[:,6]
        self.i = 0

    def append(self, i, x=None, y=None, hx=None, hy=None, ma=None, mi=None, angle=None, avg_px=None, ax=None, ay=None, ox=None, oy=None, apx=None, opx=None):
        if x is not None:
            self.data[i, 0] = x
        if y is not None:
            self.data[i, 1] = y
        if hx is not None:
            self.data[i, 2] = hx
        if hy is not None:
            self.data[i, 3] = hy
        if ma is not None:
            self.data[i, 4] = ma
        if mi is not None:
            self.data[i, 5] = mi
        if angle is not None:
            self.data[i, 6] = angle
        if avg_px is not None:
            self.data[i, 7] = avg_px
        if ax is not None:
            self.data[i, 8] = ax
        if ay is not None:
            self.data[i, 9] = ay
        if ox is not None:
            self.data[i, 10] = ox
        if oy is not None:
            self.data[i, 11] = oy
        if apx is not None:
            self.data[i, 12] = apx
        if opx is not None:
            self.data[i, 13] = opx

    def save(self, _file):
        print('Saving flytracks to {}'.format(_file))
        pd.DataFrame(columns=self.columns, data=self.data[:,:len(self.columns)]).to_csv(_file, index_label='frame')

    def trace(self,i,_len):
        return self.data[i-_len:i,0:2]
