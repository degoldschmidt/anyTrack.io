import argparse
from anytracker import Anytracker
from video import VideoCapture
import numpy as np
import pandas as pd
import cv2
import os.path as op
from tqdm import tqdm

def px(*args):
    return (int(round(arg)) for arg in args)

def main(input, nframes):
    ### create AnyTrack Tracking object
    track = Anytracker(input=input, output='output_anytrack')
    for video in track.videos:
        cap = VideoCapture(video, 0)
        w,h = cap.w, cap.h
        numframes=nframes
        writer = []
        for i, fly in enumerate(track.outdict['trajectory_files'][video]):
            _file = op.join(track.outdict['folders']['output'], 'overlay_{}_{}.avi'.format(op.basename(video).split('.')[0], i))
            print(_file)
            writer.append(cv2.VideoWriter(_file, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (600, 600), True))
        for i in tqdm(range(numframes)):
            __, frame = cap.read()
            id = 0
            for fly, roi in zip(track.outdict['trajectory_files'][video], track.outdict['ROIs'][video]):
                data=pd.read_csv(fly)
                x,y=np.array(data['head_x']), np.array(data['head_y'])
                cx, cy, cr = px(roi['x'], roi['y'], 1.5*roi['radius'])
                output = frame.copy()
                output = output[cy-cr:cy+cr,cx-cr:cx+cr]
                #output = cv2.resize(output, (tuple(px(.5*w,.5*h))))
                cv2.circle(output, tuple(px(x[i]-(cx-cr), y[i]-(cy-cr))), 1, (255,0,255),-1)
                cv2.putText(output, str(id), tuple(px(x[i]-(cx-cr)+10, y[i]-(cy-cr)+10)), cv2.FONT_HERSHEY_SIMPLEX , .5, (255,0,255), 1, cv2.LINE_AA)

                writer[id].write(cv2.resize(output, (600,600)))
                id += 1
                #cv2.imshow("Overlay", output)
                #cv2.waitKey(0) # time to wait between frames, in mSec
        cap.release()
        for w in writer:
            w.release()

if __name__ == '__main__':
    ### arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', action='store',
                        help='input file(s)/directory')
    parser.add_argument('-n', dest='nframes', action='store', type=int,
                        help='number of frames')
    args = parser.parse_args()
    input = args.input
    nframes = args.nframes
    main(input, nframes)
