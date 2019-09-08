import cv2
import numpy as np

from anytrack.video import VideoCapture

def get_background(video, num_frames=30, how='uniform', offset=0, show_all=False, frames=None):
    print('Start modelling background...', flush=True)
    cap = VideoCapture(video,0)
    h, w = cap.h, cap.w
    bgframe = np.zeros((h, w), dtype=np.float32)
    if how=='uniform':
        if frames is None:
            choices = np.random.choice(cap.len-1, num_frames)
        else:
            choices = np.random.choice(frames, num_frames)
    for i in range(num_frames):
        if how=='uniform':
            frameno = choices[i] + offset
        else:
            frameno = i + offset
        frame = cap.get_frame(frameno)
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        bgframe += img/num_frames
    step = 1
    if show_all:
        cv2.imshow("BG model iteration {}".format(step), cv2.resize(bgframe.astype(np.uint8), (600,600)))
        cv2.waitKey(0) # time to wait between frames, in mSec
        cv2.destroyAllWindows()
    newbg = np.zeros((h, w), dtype=np.float64)
    bgcount = np.zeros((h, w), dtype=np.float64)
    bgcount[:] = 1.
    step += 1
    if how=='uniform':
        if frames is None:
            choices = np.random.choice(cap.len-1, num_frames)
        else:
            choices = np.random.choice(frames, num_frames)
    for i in range(num_frames):
        if how=='uniform':
            frameno = choices[i] + offset
        else:
            frameno = i + offset
        frame = cap.get_frame(frameno)
        difference = bgframe.astype(np.float64) - cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
        __, subtr = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
        bgmask = np.zeros((h, w), dtype=np.uint8)
        bgmask[subtr==0] = frame[subtr==0, 0]
        bgcount[subtr==0] += 1.
        newbg += bgmask.astype(np.float64)
    newbg = np.divide(newbg,bgcount)
    if show_all:
        cv2.imshow("BG model iteration {}".format(step), cv2.resize(newbg.astype(np.uint8), (600,600)))
        cv2.waitKey(0) # time to wait between frames, in mSec
        cv2.destroyAllWindows()
    return newbg
