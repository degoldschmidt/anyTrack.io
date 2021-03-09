
import cv2
import numpy as np
import math
import os, platform, subprocess
import os.path as op
import uuid
from pathlib import Path
import pprint as pp
import pytrack
import argparse

#from anytrack.video import VideoCapture
from video import VideoCapture

__author__ = "Dennis Goldschmidt"
__date__ = "29.08.2018"
__email__ = "dennis.goldschmidt@neuro.fchampalimaud.org"

### TODO: this should be in core.colors
GREEN = (139, 255, 45)
MAGENTA = (217, 28, 255)
YEAST = (0, 140, 255)
SUCROSE = (0, 140, 255)
MATCH = (51,255,255)

def px(val):
    return int(round(val))

class arenaROIselector(object):
    """ Class for selecting arena ROIs manually in a given videos.
        Returns dictionary of ROIs of the four arenas ('topleft', 'topright', 'bottomleft', 'bottomright')
        containing a dictionary with the following keys:
        -   x:      x coordinate of center position in pixel
        -   y:      y coordinate of center position in pixel
        -   radius: radius of arena circle in pixel
        -   angle:  orientation of first inner yeast spot within arena circle
    """
    def __init__(self, video, setup, windowName=None, nframes=300, random=True, pattern='6_6_radial', method='supervised'):
        self.__scale = 1.6
        self.img, self.original_img = self.init_frame(video, setup, nframes, random)
        self.setup = setup
        self.__backup = self.img.copy()
        self.ROIs = []
        self.method = method
        self.selected = -1
        self.windowName = windowName
        self.__drawing = False
        self.__editing = False
        self.__automated_arenas = False
        self.loc, self.patches = [], []
        self.pattern = pattern

        if method == 'automated' or 'supervised':
            # automatic detection of arenas
            blur = cv2.GaussianBlur(self.original_img.copy()[:,:,0],(5,5),0)
            ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            self.auto_rois = contours
            for cnt in contours:
                if len(cnt) > 50:
                    (x,y),(ma,mi),angle = cv2.fitEllipse(cnt)
                    self.ROIs.append({'x': x, 'y': y, 'radius': ma/3, 'angle': 0})
            self.ROIs = sorted(self.ROIs, key=lambda k: (k['x'], k['y']))

        if method == 'supervised' or method == 'manual':
            if windowName != None:
                self.draw()
                cv2.setMouseCallback(self.windowName, self.click)
            else:
                self.windowName = "Arena ROI Selection"
                self.draw()
                cv2.setMouseCallback(self.windowName, self.click)
            self.mainloop()

    def init_frame(self, video, setup, nframes, random, start_frame=0):
        cap = VideoCapture(video,start_frame)
        ret, image = cap.read()
        self.__width, self.__height = int(image.shape[1]/self.__scale), int(image.shape[0]/self.__scale)
        image = cap.get_average(100, random=True, to=(self.__width, self.__height))
        original = image.copy()
        w, h = image.shape[0], image.shape[1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, 'Video: {}'.format(video), (20, h-35), font, .5, GREEN, 1, cv2.LINE_AA)
        cv2.putText(image, 'Setup: {}'.format(setup), (20, h-10), font, .5, GREEN, 1, cv2.LINE_AA)
        return image, original

    def mainloop(self):
        while True:
            k = cv2.waitKeyEx(1)
            if k == ord('r'):
                self.resetCanvas()
            elif k == ord('1'):
                self.select('topleft')
            elif k == ord('2'):
                self.select('topright')
            elif k == ord('3'):
                self.select('bottomleft')
            elif k == ord('4'):
                self.select('bottomright')
            elif k == 27:
                break
            elif k == 65535:
                self.delete()
            elif k == ord('c'):
                self.__automated_arenas = not self.__automated_arenas
                self.draw()
            elif k == 43 or k == ord('w'):
                self.scale_by(1)
            elif k == 45 or k == ord('s'):
                self.scale_by(-1)
            elif k == ord('d'):
                self.rotate_by(1)
            elif k == ord('a'):
                self.rotate_by(-1)
            elif k == 32:
                self.save_templates()
            elif k == 13:
                self.loc = self.match_templates('arena', self.setup)
                self.patches = self.get_peak_matches()
                self.draw()
            elif k == -1:
                continue
            else:
                print(k)

    def resetCanvas(self):
        """Function to reset the canvas with the given image. This resets the current ROI in memory, but leaves the entire ROI list untouched.
        Call this function from the mainloop selectively by listening to specific key strokes."""
        self.img = self.__backup.copy()
        self.ROIs = []
        self.selected = 'topleft'
        self.draw()


    def click(self, event, x, y, flags, param):
        """Main click event for the mouse. Allowed actions:
        Left click: If a ROI is open, that is, it is not enclosed, it adds another point, where the mouse clicked to the polygon
        Right click: If the ROI is open, then it closes the ROI polygon. This was done to make sure that the ROI is closed, since
                     even a pixel of open ROI, while invisible to the human eye, might wreak havoc for algorithms like flood-fill
                     and would need further prepocessing (Trust me, I faced it and this solves a bit of the headaches). Updates
                     the ROI list automatically when the polygon is closed. If the ROI list is empty, then you probably didn't close
                     the polygon. Try right clicking next time.
        Alternative Right click: This is triggered only when the polygon is closed. This starts the orientation mode, recognizable
                     by a line following the mouse from the centroid of the ROI. This mode is used to specify a guide from which
                     the orientation of the ROI is to be calculated.
        Alternative Left click: This is triggered only when the polygon is closed and the orientation mode has started. This
                     finalizes the orientation to face in the direction where the user clicks. So you get a line from which the
                     orientation of the ROI is estimated. The model assumes an upright, right handed, 360 degrees rotational frame.
                     """
        if event == cv2.EVENT_LBUTTONDBLCLK:
            for pt, w, v in self.patches:
                x0, y0 = pt
                if np.sqrt((x0-x)**2 + (y0-y)**2) < w:
                    self.ROIs[self.selected]['x'] = x0
                    self.ROIs[self.selected]['y'] = y0
                    self.ROIs[self.selected]['radius'] = w/2
                    self.ROIs[self.selected]['angle'] = 0
                    self.__editing = False
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.__editing = False
            for id, arena in enumerate(self.ROIs):
                if arena is not None:
                    x0, y0 = arena['x'], arena['y']
                    if np.sqrt((x0-x)**2 + (y0-y)**2) < arena['radius']:
                        self.selected = id
                        self.__editing = True
                    elif self.selected == id:
                        self.next_selected()
            for pt, w, v in self.patches:
                x0, y0 = pt
                if np.sqrt((x0-x)**2 + (y0-y)**2) < w:
                    self.__editing = False
            if not self.__editing:
                self.__drawing = True
                self.x0, self.y0 = x, y
                self.ROIs.append({})
                self.ROIs[-1]['x'] = x
                self.ROIs[-1]['y'] = y
                self.ROIs[-1]['radius'] = 100
                self.ROIs[-1]['angle'] = 0
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.__drawing:
                s = self.ROIs[self.selected]['radius']
                x0, y0 = self.ROIs[self.selected]['x'], self.ROIs[self.selected]['y']
                self.ROIs[self.selected]['radius'] = max(1, s+0.01*(x-x0))
                self.ROIs[self.selected]['angle'] = math.fmod(0.01*(y-y0), 2.*np.pi)
            elif self.__editing:
                self.ROIs[self.selected]['x'] = x
                self.ROIs[self.selected]['y'] = y
        elif event == cv2.EVENT_LBUTTONUP:
            if self.__drawing:
                self.__drawing = False
            elif self.__editing:
                self.__editing = False
        self.draw()

    def delete(self):
        self.ROIs.pop(self.selected)
        self.selected = len(self.ROIs)-1
        self.draw()

    def draw(self):
        img = self.img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        ### draw arenas
        for id, arena in enumerate(self.ROIs):
            if arena is not None:
                ### draw arenas
                x, y, r, a  = int(arena['x']), int(arena['y']), int(arena['radius']), arena['angle']
                if id == self.selected:
                    color = MAGENTA
                    lw = 2
                else:
                    color = GREEN
                    lw = 1
                cv2.circle(img,(x, y),r,color,lw)
                cv2.circle(img,(x, y),1,color,-1)
                cv2.rectangle(img, (px(x-1.5*r),px(y-1.5*r)), (px(x+1.5*r), px(y+1.5*r)), color, 1)
                ### draw spots
                if self.pattern == '6_6_radial':
                    for j in range(2):
                        for i in range(3):
                            sr = 1
                            if i==0 and j==0:
                                sr = 2
                            spot_r, angle = (j+1)*r/2.5, a+i*(2*np.pi/3)-j*(2*np.pi/12)
                            sx, sy = int(x + spot_r*np.cos(angle)), int(y + spot_r*np.sin(angle))
                            cv2.circle(img, (sx, sy), int(3*r/50),YEAST, sr)
                cv2.putText(img, str(id), (x+10, y-10), font, .5, color, 1, cv2.LINE_AA)
                cv2.putText(img, str(r), (x+10, y+10), font, .5, color, 1, cv2.LINE_AA)
        ### draw labels
        cv2.putText(img, 'Selected arena: {}'.format(self.selected), (20, 30), font, .75, (139, 255, 45), 1, cv2.LINE_AA)

        for pt, w, v in self.patches:
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + w), MATCH, 1)
            cv2.circle(img, (int(pt[0] + w/2), int(pt[1] + w/2)), int(w/2),MATCH, 1)

        if self.__automated_arenas:
            contours = self.auto_rois
            cv2.drawContours(img, contours, -1, (0,255,0), 3)
            for cnt in contours:
                if len(cnt) > 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (x,y),(ma,mi),angle = ellipse
                    cv2.ellipse(img,ellipse,(0,255,0),2)
                    cv2.circle(img, (px(x), px(y)), px(ma/3), MATCH, 1)


        cv2.imshow(self.windowName, cv2.resize(img, (700,700)))

    def next_selected(self):
        self.selected = len(self.ROIs)

    def get_spots(self, id):
        x, y, r, a = self.ROIs[id]['x']* self.__scale, self.ROIs[id]['y']* self.__scale, self.ROIs[id]['radius']* self.__scale, self.ROIs[id]['angle']
        a = a%(2.*np.pi/3)
        spots = []
        if self.pattern == '6_6_radial':
            for j in range(2):
                for i in range(3):
                    ### yeast spot
                    spot_id = 2*(i+j*3)
                    spot_r, angle = (j+1)*r/2.5, a+i*(2*np.pi/3)-j*(2*np.pi/12)
                    sx, sy = x + spot_r*np.cos(angle), y + spot_r*np.sin(angle)
                    spots.append({'id': spot_id, 'x': round(float(sx),1), 'y': round(float(sy),1), 'substrate': 'yeast'})
                    ### sucrose spot
                    spot_id = 2*(i+j*3)+1
                    spot_r, angle = (j+1)*r/2.5, a+i*(2*np.pi/3)-j*(2*np.pi/12)+np.pi/3
                    sx, sy = x + spot_r*np.cos(angle), y + spot_r*np.sin(angle)
                    spots.append({'id': spot_id, 'x': round(float(sx),1), 'y': round(float(sy),1), 'substrate': 'sucrose'})
        return spots

    def rotate_by(self, val):
        if self.ROIs[self.selected] is not None:
            self.ROIs[self.selected]['angle'] += 0.1*val/(2*np.pi)
            self.ROIs[self.selected]['angle'] = math.fmod(self.ROIs[self.selected]['angle'], 2*np.pi)
            self.draw()

    def get_peak_matches(self):
        patches = []
        maxv = []
        for pt, w, v in self.loc:
            if len(patches) == 0:
                patches.append((pt, w, v))
                maxv = v
            else:
                outside = len(patches)*[False]
                for j, patch in enumerate(patches):
                    x, y = patch[0][0], patch[0][1]
                    if abs(x-pt[0]) < w and abs(y-pt[1]) < w:
                        if v > maxv:
                            patches[j] = (pt, w, v)
                            maxv = v
                    elif abs(x-pt[0]) > w or abs(y-pt[1]) > w:
                        outside[j] = True
                if all(outside):
                    patches.append((pt, w, v))
                    maxv = v
        to_delete = []
        for i, (pt, w, v) in enumerate(patches):
            for id, arena in self.arenas.items():
                if arena is not None:
                    if np.sqrt( (pt[0]+w/2 - arena['x'])**2 + (pt[1]+w/2 - arena['y'])**2 ) < w:
                        to_delete.append(i)
        for i in sorted(to_delete, reverse=True):
            del patches[i]
        return patches

    def get_templates_folder(self):
        if platform.system() == 'Darwin':
            return '/Users/degoldschmidt/.pytrack/templates/{}'.format(self.__id)
        else:
            return '/home/degoldschmidt/.pytrack/templates/{}'.format(self.__id)

    def scale_by(self, val):
        if self.ROIs[self.selected] is not None:
            self.ROIs[self.selected]['radius'] += val
            if self.ROIs[self.selected]['radius'] < 1:
                self.ROIs[self.selected]['radius'] = 1
            self.draw()

    def select(self, arenaid):
        self.selected = arenaid
        self.draw()

    def get(self):
        rois = []
        for id, roi in enumerate(self.ROIs):
            if roi is None:
                pass
            else:
                rois.append({})
                rois[id]['x'] = round(roi['x'] * self.__scale,1)
                rois[id]['y'] = round(roi['y'] * self.__scale,1)
                rois[id]['radius'] = round(float(roi['radius'] * self.__scale),1)
                rois[id]['outer'] = round(float(1.25 * roi['radius'] * self.__scale),1)
                rois[id]['scale'] = round(float((roi['radius']* self.__scale)/25.),3)
                rois[id]['id'] = id
                rois[id]['spots'] = self.get_spots(id)
        return rois

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()
    input = args.input
    if op.isdir(input):
        videos = [file for file in sorted(os.listdir(input)) if file.endswith('.avi')]
    elif op.isfile(input) and input.endswith('.avi'):
        videos = [args.input]
        print(cv2.__version__)
    for video in videos:
        selector = arenaROIselector(video, 'cam01', pattern=None)
        rois = selector.get()
        print(rois)
        cv2.destroyAllWindows()
    """
    video = '/media/degoldschmidt/DATA_DENNIS_002/opto_oloop_2019-07-03T15_49_51.avi'
    #folder = '/media/degoldschmidt/DATA_DENNIS_002/backup/2018_08_01/'
    #videos = [op.join(folder, video) for video in os.listdir(folder) if video.endswith('avi') and video.startswith('cam01')]
    #print(videos)
    if platform.system() == 'Darwin':
        video = '/Users/degoldschmidt/Desktop/tracking_test_data/cam01_2017-11-24T08_26_19.avi'
    #for video in videos:
    selector = arenaROIselector(video, 'cam01', pattern=None)
    rois = selector.get()
    print(rois)
    cv2.destroyAllWindows()
    """
