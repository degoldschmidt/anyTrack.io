import cv2
from background_subtraction import BackgroundSubtraction

class VideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.grabbed, self.frame = self.vid.read()
        # Get video source width and height
        self.start=0
        self.len = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aratio = self.width/self.height

        self.bgsub = BackgroundSubtraction(self)

    def get_background(self):
        return True, self.bgsub.bg

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get(self, var):
        return self.vid.get(var)

    def set(self, var1, var2):
        self.vid.set(var1, var2)

    def set_frame(self, var):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, var)

    def skip(self, frames):
        if self.get(cv2.CAP_PROP_POS_FRAMES)+frames < self.len:
            self.set_frame(self.get(cv2.CAP_PROP_POS_FRAMES)+frames)

    def read(self):
        self.grabbed, self.frame = self.vid.read()
        frame = self.frame.copy()
        grabbed = self.grabbed
        return grabbed, frame

    def restart(self):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.start)

    def rewind(self, frames):
        if self.get(cv2.CAP_PROP_POS_FRAMES)+frames >= 0:
            self.set_frame(self.get(cv2.CAP_PROP_POS_FRAMES)-frames)

    def stop(self):
        self.vid.release()

    def tracking(self):
        return True, self.bgsub.update()

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
