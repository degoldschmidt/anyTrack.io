import cv2
import numpy as np

class VideoCapture:
    def __init__(self, src, startframe):
        self.src = src
        self.cap = cv2.VideoCapture(self.src,0)
        self.start = startframe
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, startframe)
        self.grabbed, self.frame = self.cap.read()
        self.len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
        self.duration = self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000. ### in secs
        self.cap.set(cv2.CAP_PROP_POS_AVI_RATIO,0)
        self.w  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_average(self, nframes, random=False, to=None):
        avg = np.zeros(self.frame.shape, dtype=np.float32)
        if random:
            choices = np.random.choice(self.len-self.start, nframes)
        for i in range(nframes):
            if random:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, choices[i]+self.start)
            ret, frame = self.read()
            cv2.accumulate(frame,avg)
            img = cv2.convertScaleAbs(avg, alpha=1./nframes)
        if to is not None:
            img = cv2.resize(img, to)
        return img

    def get(self, var):
        return self.cap.get(var)

    def get_frame(self, var):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, var)
        _, frame = self.read()
        return frame

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def set_frame(self, var):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, var)

    def read(self):
        self.grabbed, self.frame = self.cap.read()
        frame = self.frame.copy()
        grabbed = self.grabbed
        return grabbed, frame

    def restart(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)

    def release(self):
        self.cap.release()
