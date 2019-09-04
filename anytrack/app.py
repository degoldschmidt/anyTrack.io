import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
import cv2
import PIL.Image, PIL.ImageTk
import video_capture as vc

class VideoPlayer(ttk.Frame):
    def __init__(self, master=None, mode_var=None, video=None):
        ttk.Frame.__init__(self, master)
        self.video = None##vc.VideoCapture(self.video_source)
        self.__mode = mode_var
        self.w, self.h = 500, 500
        self.canvas = tk.Canvas(self, width = self.w, height = self.h, bg='#393939')
        self.state = 'play'
        self.__packing()

    def __packing(self):
        self.canvas.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.rowconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)

    def update(self):
        if self.video is not None:
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
                ret, (frame, thr) = video.subtract()
                if ret:
                    self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.resize(frame.astype(np.uint8), (self.w, self.h))))
                    self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            elif self.state == 'contours':
                self.__mode.set('{:06d}/{:06d}'.format(int(video.get(cv2.CAP_PROP_POS_FRAMES)), video.len))
                ret, frame = video.get_frame()
                ret, (frame2, thr) = video.subtract()
                contours = video.track(thr)
                cv2.drawContours(frame, contours, -1, (0,255,0), 1)
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
    def __init__(self, master=None, model=None, videos=None):
        self.master = master
        master.title("AnyTrack v1.0")
        master.geometry("1200x1000")
        ttk.Frame.__init__(self, master)
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
        elif event.keysym == 'c':
            self.__video.state = 'contours'
        elif event.keysym == 'Right':
            self.vid.skip(1000)
        elif event.keysym == 'Left':
            self.vid.rewind(1000)
        else:
            print(event.keysym)

    def __init(self, model):
        self.__video = VideoPlayer(self)
        self.__label  = LabelFrm(self)
        self.__info = ttk.Frame(self)
        self.__treeview = ttk.Frame(self)

        # create the tree and scrollbars
        self.dataCols = ('country', 'capital')
        self.tree = ttk.Treeview(columns=self.dataCols,
                                 show = 'headings')

        ysb = ttk.Scrollbar(orient=tk.VERTICAL, command= self.tree.yview)
        xsb = ttk.Scrollbar(orient=tk.HORIZONTAL, command= self.tree.xview)
        self.tree['yscroll'] = ysb.set
        self.tree['xscroll'] = xsb.set

        # add tree and scrollbars to frame
        self.tree.grid(in_=self.__treeview, row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        ysb.grid(in_=self.__treeview, row=0, column=1, sticky=tk.N+tk.S)
        xsb.grid(in_=self.__treeview, row=1, column=0, sticky=tk.E+tk.W)

        # set frame resize priorities
        self.__treeview.rowconfigure(0, weight=1)
        self.__treeview.columnconfigure(0, weight=1)
        self.__packing()

    def __packing(self):
        tk.Grid.rowconfigure(self.master, 0, weight=1)
        tk.Grid.columnconfigure(self.master, 0, weight=1)
        self.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self.__video.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.__label.grid(row=1, column=0, sticky=tk.N+tk.S+tk.E+tk.W)

        self.__treeview.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        self.__info.grid(row=1, column=1, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.rowconfigure(self, 0, weight=1)
        tk.Grid.rowconfigure(self, 1, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 1, weight=1)

    def __update(self):
        self.master.after(self.delay, self.__update)

def main():
    root = tk.Tk()
    app = AnytrackApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
