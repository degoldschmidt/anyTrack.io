import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import cv2
import PIL.Image, PIL.ImageTk
import video_capture as vc

def get_screen_dims(master, video, update=False):
    if update:
        _height = master.winfo_height()
        _width = master.winfo_width()
        if _height < 100 or _width <100:
            _height, _width = 900, 800
        w, h = min(_width, video.width), min(_height, video.height)
    else:
        _width = master.winfo_screenwidth()
        _height = master.winfo_screenheight()
        w, h = min(2*_width/3, video.width), min(2*_height/3, video.height)
    if video.aratio > w/h:
        h = w/video.aratio
    else:
        w = h*video.aratio
    return int(w), int(h)

class Model():
    def __init__(self):
        self.mode = tk.StringVar()

class VideoPlayer(ttk.Frame):
    def __init__(self, master=None, mode_var=None, video=None):
        ttk.Frame.__init__(self, master)
        self.__mode = mode_var
        self.w, self.h = get_screen_dims(master, video)
        self.canvas = tk.Canvas(self, width = self.w, height = self.h)
        self.state = 'play'
        self.__packing()

    def __packing(self):
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.rowconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)

    def update(self, video):
        if self.state == 'play':
            self.__mode.set('{:06d}/{:06d}'.format(int(video.get(cv2.CAP_PROP_POS_FRAMES)), video.len))
            ret, frame = video.get_frame()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(frame, (self.w, self.h))))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
        elif self.state == 'background':
            self.__mode.set('Background frame')
            ret, frame = video.get_background()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(frame.astype(np.uint8), (self.w, self.h))))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.state = 'idle'
        elif self.state == 'tracking':
            self.__mode.set('{:06d}/{:06d}'.format(int(video.get(cv2.CAP_PROP_POS_FRAMES)), video.len))
            ret, (frame, thr) = video.tracking()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(frame.astype(np.uint8), (self.w, self.h))))
                self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)


class LabelFrm(ttk.Frame):
    def __init__(self, master=None, mode_var=None):
        ttk.Frame.__init__(self, master)
        self.__mode = mode_var
        self.__init()

    def __init(self):
        self.__label = ttk.Label(self, textvariable=self.__mode)
        self.__packing()

    def __packing(self):
        self.__label.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.rowconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)

class AnytrackApp(ttk.Frame):
    def __init__(self, master=None, model=None):
        self.master = master
        master.title("AnyTrack v1.0")
        master.geometry("800x900")
        ttk.Frame.__init__(self, master)
        #self.video_source = '/Volumes/DashDATA/working_data/2019_04_23/cam01_2019-04-23T10_32_40.avi'
        self.video_source = '/home/degoldschmidt/Desktop/example_data/cam01_2018-08-09T15_16_20.avi'
        self.vid = vc.VideoCapture(self.video_source)
        self.__init(model)
        self.delay = 20
        self.__bind()
        self.__update()
        master.mainloop()

    def __bind(self):
        self.master.bind("<KeyPress>", self.__command)

    def __command(self, event):
        if event.keysym == 'Escape':
            self.master.destroy()
        elif event.keysym == 'b':
            self.__video.state = 'background'
        elif event.keysym == 't':
            self.__video.state = 'tracking'
            self.vid.restart()
        elif event.keysym == 'Right':
            self.vid.skip(1000)
        elif event.keysym == 'Left':
            self.vid.rewind(1000)
        else:
            print(event.keysym)

    def __init(self, model):
        self.__model  = Model()
        self.__video = VideoPlayer(self, self.__model.mode, self.vid)
        self.__label  = LabelFrm(self,  self.__model.mode)
        self.__packing()

    def __packing(self):
        tk.Grid.rowconfigure(self.master, 0, weight=1)
        tk.Grid.columnconfigure(self.master, 0, weight=1)
        self.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self.__video.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.__label.grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.rowconfigure(self, 0, weight=1)
        tk.Grid.rowconfigure(self, 1, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)

    def __update(self):
        self.__video.w, self.__video.h = get_screen_dims(self.master, self.vid, update=True)
        self.__video.update(self.vid)
        self.master.after(self.delay, self.__update)

def main():
    root = tk.Tk()
    model = Model()
    app = AnytrackApp(root, model)
    root.mainloop()

if __name__ == '__main__':
    main()
