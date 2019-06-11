import numpy as np
import cv2
import video_capture as vc

class BackgroundSubtraction:
    def __init__(self, video, background_frames=10, adaptation_rate=0., threshold_value=(30, 15), threshold_type=cv2.THRESH_BINARY, subtraction_method='Dark'):
        ### Set up video capture
        self.cap = video
        ### Background subtraction parameters
        self.__bgframes = background_frames
        self.__adaptrate = adaptation_rate
        self.__thrval = threshold_value[0]
        self.__wthrval = threshold_value[1]
        self.__thr = threshold_type
        self.__method = subtraction_method
        if self.__adaptrate == 0:
            self.bg = self.get_background(random=True)
        else:
            self.bg = self.cap.frame
        self.cap.set_frame(0)

    def get_background(self, random=True):
        bgframe = np.zeros(self.cap.frame.shape, dtype=np.float32)
        nframes = self.cap.len - self.cap.start
        if random:
            choices = np.random.choice(nframes, self.__bgframes)
        for i in range(self.__bgframes):
            if random:
                frameno = choices[i]
                self.cap.set_frame(choices[i] + self.cap.start)
            __, img = self.cap.read()
            bgframe += img
            if i == self.__bgframes-1:
                bgframe /= self.__bgframes
        newbg = np.zeros(self.cap.frame.shape, dtype=np.float64)
        bgcount = np.zeros(self.cap.frame.shape, dtype=np.float64)
        skip = 20000
        for i in range(0,self.cap.len,skip):
            self.cap.set_frame(i)
            _, frame = self.cap.get_frame()
            difference = bgframe[:,:,0].astype(np.float64) - cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).astype(np.float64)
            __, subtr = cv2.threshold(difference, 20, 255, cv2.THRESH_BINARY)
            bgmask = np.zeros(frame.shape, dtype=np.uint8)
            bgmask[subtr==0] = frame[subtr==0]
            bgcount[subtr==0] += 1.
            newbg += bgmask.astype(np.float64)
        bgcount[bgcount==0] = 1.
        newbg = np.divide(newbg,bgcount)
        bgframe = newbg[:,:,0]
        self.bg = bgframe
        return bgframe

    def update(self):
        #if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.len:
        #    self.cap.set_frame(0)
        print(self.cap.get(cv2.CAP_PROP_POS_FRAMES), self.cap.len)
        ### grab frame
        ret, frame = self.cap.read()
        ### background subtraction
        #img = frame.copy()
        if self.__method == 'Bright':
            difference = frame[:,:,0] - self.bg
        elif self.__method == 'Dark':
            difference = self.bg - frame[:,:,0]
        else:
            difference = np.abs(frame[:,:,0] - self.bg)
        #if self.__adaptrate > 0:
        #    self.bg = self.__adaptrate * image + (1 - self.__adaptrate) * self.bg
        __, strong_sub = cv2.threshold(difference, self.__thrval, 255, self.__thr)
        #__, weak_sub = cv2.threshold(difference, self.__wthrval, 255, self.__thr)
        return frame, strong_sub
