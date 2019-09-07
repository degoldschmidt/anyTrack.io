from __future__ import print_function, unicode_literals
import os, sys
import os.path as op
import cv2
import argparse
from pprint import pprint
import numpy as np
import pandas as pd
import io, yaml
from tqdm import tqdm
import threading
import matplotlib.pyplot as plt

from cli import checkbox
from roiselect import arenaROIselector
from video import VideoCapture

def px(val):
    return int(round(val))

def dist(pos1, pos2):
    dx, dy = (pos1[0] - pos2[0]), (pos1[1] - pos2[1])
    return np.sqrt(dx * dx + dy * dy)

def read_yaml(_file):
    """ Returns a dict of a YAML-readable file '_file'. Returns None, if file is empty. """
    with open(_file, 'r') as stream:
        out = yaml.load(stream, Loader=yaml.SafeLoader)
    return out

def write_yaml(_file, _dict):
    """ Writes a given dictionary '_dict' into file '_file' in YAML format. Uses UTF8 encoding and no default flow style. """
    with io.open(_file, 'w+', encoding='utf8') as outfile:
        yaml.dump(_dict, outfile, default_flow_style=False, allow_unicode=True)

colors = [  (255, 151, 42),
            (42, 127, 255),
            (8, 204, 34),
            (168, 42, 255),
            (255, 68, 42),
            (233, 0, 154),
            (245, 255, 47),
            (34, 88, 3),
            (240, 240, 240),
            (120, 88, 170)]

def get_contours(frame, background, threshold_level=10, thresholding='dark'):
    bg = background.astype(np.float64)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if thresholding == 'dark':
        difference =  bg - gray_frame
    elif thresholding == 'bright':
        difference =  gray_freame - bg
    else:
        difference =  np.abs(bg - gray_frame)
    __, subtr = cv2.threshold(difference, threshold_level, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(subtr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_contour_stats(video, background, num_frames=500, how='uniform'):
    print('Infer contour statistics...', flush=True)
    cap = VideoCapture(video,0)
    h, w = cap.h, cap.w
    number_contours = np.zeros(num_frames)
    all_areas = []
    if how=='uniform':
        choices = np.random.choice(cap.len-1, num_frames)
    for i in range(num_frames):
        if how=='uniform':
            frameno = choices[i]
        else:
            frameno = i
        frame = cap.get_frame(frameno)
        contours = get_contours(frame, background)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20]
        number_contours[i] = len(contours)
        for cnt in contours:
            all_areas.append(cv2.contourArea(cnt))
    mean_area = np.nanmean(all_areas)
    mean_number = np.nanmean(number_contours)
    area_contours = np.zeros((num_frames, int(round(mean_number))))
    for i in range(num_frames):
        if how=='uniform':
            frameno = choices[i]
        else:
            frameno = i
        frame = cap.get_frame(frameno)
        contours = get_contours(frame, background)
        areas = sorted([cv2.contourArea(cnt) for cnt in contours], key=lambda x: np.abs(x-mean_area))[:int(round(mean_number))]
        area_contours[i, :] = np.array(areas)
    return mean_number, np.amin(area_contours), np.amax(area_contours)

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

def get_videos(input, output):
    if 'videos' in output:
        videos = [_file for _file in output['videos']]
        dir = output['directory']
    else:
        if op.isfile(input):
            dir = op.dirname(input)
            videos = [input]
        elif op.isdir(input):
            dir = input
            videos = [op.join(input, _file) for _file in os.listdir(input) if _file.endswith('avi')]
        else:
            raise FileNotFoundError
        videos = [op.join(input, _file) for _file in checkbox('videos', videos, msg='Select videos for tracking:')]
    return videos, dir

def load_trajectory(_file):
    df = pd.read_csv(_file)
    size = df.shape[0]
    trajectory = Trajectory(size)
    for i,col in enumerate(trajectory.columns):
        trajectory.data[:,i] = np.array(df[col])
    return trajectory

def run_bg_subtraction(video, background=None, threshold_level=10, thresholding='dark', show=0, n_contours=4, min_size=150, max_size=250, rois=None):
    cap = VideoCapture(video,0)
    flytracks = [FlyTrajectory(cap.len) for i in range(n_contours)]
    flag = False
    for frameno in tqdm(range(cap.len)):
        frame = cap.get_frame(frameno)
        contours = get_contours(frame, background, threshold_level=threshold_level, thresholding=thresholding)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt)>.9*min_size and cv2.contourArea(cnt)<1.1*max_size]
        output = frame.copy()
        while len(contours) > n_contours:
            if frameno > 0:
                min_dists = [np.amin([dist((flytrack.x[frameno-1],flytrack.y[frameno-1]), cv2.fitEllipse(cnt)[0]) for flytrack in flytracks]) for cnt in contours]
                print(min_dists)
                popidx = min_dists.index(max(min_dists))
            else:
                areas = [cv2.contourArea(cnt) for cnt in contours]
                print(areas)
                popidx = areas.index(max(areas))
            print(popidx)
            contours.pop(popidx)
            raise IndexError
        if len(contours) == n_contours:
            order = []
            if frameno == 0:
                order = [i for i in range(n_contours)]
            for j,cnt in enumerate(contours):
                if frameno > 0:
                    min_dists = [dist((flytrack.x[frameno-1],flytrack.y[frameno-1]), cv2.fitEllipse(cnt)[0]) for flytrack in flytracks]
                    idx = min_dists.index(min(min_dists))
                    order.append(idx)
            if len(order) != len(set(order)):
                flag = True
            for j in order:
                ellipse = cv2.fitEllipse(contours[j])
                (x,y),(ma,mi),a = ellipse
                flytracks[j].append(x=x,y=y,ma=ma,mi=mi,angle=a)
                cv2.ellipse(output,ellipse,(0,255,0),2)
                cv2.circle(output,(px(x),px(y)), 1, (255,0,255),1)
                pts = flytracks[j].data[frameno-30:frameno,0:2].astype(np.int32)
                ##print(pts.shape, pts)
                pts = pts.reshape((-1,1,2))
                cv2.polylines(output,[pts],False,(0,255,255))
                cv2.putText(output, '{}'.format(j), (px(x)+10, px(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1, cv2.LINE_AA)
        if show>0 or flag:
            if frameno%show==0 or flag:
                cv2.imshow("Tracking", cv2.resize(output, (700,700)))
                if flag:
                    cv2.waitKey(0) # time to wait between frames, in mSec
                    flag = False
                else:
                    cv2.waitKey(1) # time to wait between frames, in mSec

    return flytracks

class FlyTrajectory(object):
    def __init__(self, num_frames):
        self.columns = ['body_x', 'body_y', 'head_x', 'head_y', 'major', 'minor', 'angle']
        self.data = np.zeros((num_frames,len(self.columns)))
        self.data[:] = np.nan
        self.x = self.data[:,0]
        self.y = self.data[:,1]
        self.hx = self.data[:,2]
        self.hy = self.data[:,3]
        self.major = self.data[:,4]
        self.minor = self.data[:,5]
        self.angle = self.data[:,6]
        self.i = 0

    def append(self, x=None, y=None, hx=None, hy=None, ma=None, mi=None, angle=None):
        if x is not None:
            self.data[self.i, 0] = x
        if y is not None:
            self.data[self.i, 1] = y
        if hx is not None:
            self.data[self.i, 2] = hx
        if hy is not None:
            self.data[self.i, 3] = hy
        if ma is not None:
            self.data[self.i, 4] = ma
        if mi is not None:
            self.data[self.i, 5] = mi
        if angle is not None:
            self.data[self.i, 6] = angle
        self.i += 1


    def reset(self):
        self.i=0

    def trace(self,_len):
        return self.data[self.i-_len:self.i,0:2]

class Tracking(object):
    def __init__(self, input=None, output=None):
        ### create output folder inside input folder
        outfolder = op.join(input, output)
        os.makedirs(outfolder, exist_ok=True)
        outdict_file = op.join(outfolder, 'outdict.yaml')
        ### config file for multiple runs
        if op.isfile(outdict_file):
            outdict = read_yaml(outdict_file)
        else:
            outdict = {}
        if 'folders' not in outdict:
            outdict['directory'] = input
            outdict['folders'] = {}
            outdict['folders']['background'] = op.join(input, output, 'bg')
            outdict['folders']['image_stats'] = op.join(input, output, 'image_stats')
            outdict['folders']['output'] = outfolder
            for f in outdict['folders']:
                os.makedirs(outdict['folders'][f], exist_ok=True)
        ### getting videos based on input
        self.videos, basedir = get_videos(input, outdict)
        outdict['videos'] = self.videos
        self.outdict=outdict
        self.outdict_file = outdict_file
        self.background = {}

    def infer(self):
        print('Infer contour statistics...', flush=True)
        if 'number_contours' not in self.outdict:
            self.outdict['number_contours'] = {}
            self.outdict['min_size'] = {}
            self.outdict['max_size'] = {}
        for video in tqdm(self.videos):
            if video in self.outdict['number_contours']:
                numcnts, min_size, max_size = self.outdict['number_contours'][video], self.outdict['min_size'][video], self.outdict['max_size'][video]
            else:
                numcnts, min_size, max_size = get_contour_stats(video, self.background[video])
                self.outdict['number_contours'][video] = int(round(numcnts))
                self.outdict['min_size'][video] = float(min_size)
                self.outdict['max_size'][video] = float(max_size)
            print('avg.number of contours:\t{}\nminimum area:\t{}\nmaximum area:\t{}'.format(numcnts, min_size, max_size))
        return self.outdict['number_contours'], self.outdict['min_size'], self.outdict['max_size']

    def image_stats(self, skip=30):
        print('Analyze image statistics...')
        avgs={}
        for video in tqdm(self.videos):
            if 'image_stats_files' not in self.outdict:
                self.outdict['image_stats_files'] = {}
            if video in self.outdict['image_stats_files']:                      ### load image stats
                avgs[video] = np.load(self.outdict['image_stats_files'][video])
            else:                                                               ### run image stats
                cap = VideoCapture(video, 0)
                average_intensity = np.zeros(cap.len)
                for i in tqdm(range(int(len(average_intensity)/skip))):
                    cap.set_frame(i*skip)
                    ret, image = cap.read()
                    average_intensity[i*skip] = np.nanmean(image)
                mu=np.mean(average_intensity[average_intensity>0.])
                std=np.std(average_intensity[average_intensity>0.])
                for i in tqdm(range(len(average_intensity))):
                    if i%30!=0:
                        if average_intensity[int(i/skip)*skip] > mu+std:
                            average_intensity[i] = average_intensity[int(i/skip)*skip]
                        if (int(i/skip)+1)*skip < len(average_intensity):
                            if average_intensity[(int(i/skip)+1)*skip] > mu+std:
                                average_intensity[i] = average_intensity[(int(i/skip)+1)*skip]
                binary_vec = np.zeros(average_intensity.shape, dtype=np.uint8)
                binary_vec[average_intensity>=mu+std] = 1
                outfile = op.join(self.outdict['folders']['image_stats'], '{}_stats.npy'.format(op.basename(video).split('.')[0]))
                self.outdict['image_stats_files'][video] = outfile
                np.save(outfile, binary_vec)
                avgs[video] = binary_vec
        return avgs

    def model_bg(self, baselines=None):
        print('Modelling background...', flush=True)
        if 'background_files' not in self.outdict:
            self.outdict['background_files'] = {}
        for video in tqdm(self.videos):
            if video not in self.outdict['background_files']:
                self.outdict['background_files'][video] = op.join(self.outdict['folders']['background'], '{}_bg.png'.format(op.basename(video).split('.')[0]))
            if op.isfile(self.outdict['background_files'][video]):
                print('Loading background file...', end='', flush=True)
                self.background[video] = cv2.imread(self.outdict['background_files'][video])[:,:,0]
                print('[DONE]')
            else:
                if baselines is not None:
                    only_these = np.where(baselines[video]==0)[0]
                else:
                    only_these = None
                self.background[video] = get_background(video, show_all=True, frames=only_these)
                cv2.imwrite(self.outdict['background_files'][video], self.background[video])

    def run(self, threshold_level=10, thresholding='dark', use_threads=1):
        print('Run tracking...', flush=True)
        for video in tqdm(self.videos):
            run_bg_subtraction(video, background=self.background[video], threshold_level=threshold_level, thresholding=thresholding, n_contours=self.outdict['number_contours'][video], min_size=self.outdict['min_size'][video], max_size=self.outdict['max_size'][video])


    def roi_select(self, method='automated'):
        print('Initialize ROI selector...')
        self.all_rois = {}
        if 'ROIs' in self.outdict:
            self.all_rois = self.outdict['ROIs']
        for video in tqdm(self.videos):
            if video not in self.all_rois:
                selector = arenaROIselector(video, 'cam05', method=method, pattern=None)
                rois = selector.get()
                self.all_rois[video] = rois
                cv2.destroyAllWindows()
        self.outdict['ROIs'] = self.all_rois
        return self.all_rois


def main():
    ### arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', action='store',
                        help='input file(s)/directory')
    args = parser.parse_args()
    input = args.input

    ### create AnyTrack Tracking object
    track = Tracking(input=input, output='output_anytrack')
    ### step 1: batch run ROIfinder to get arenas (automated/supervised/manual) recommended to use 'supervised'
    track.roi_select(method='supervised')

    ### step 2: get image statistics (average signal over time, )
    avg_pxs = track.image_stats(skip=30)

    ### step 3: background modelling (DONE)
    track.model_bg(baselines=avg_pxs)

    ### step 4: infer contour statistics (DONE)
    track.infer()
    try:
        ### step 5: background subtraction & contour matching & centroid fit & identity (DONE, frame -> contours -> centroid+pixelinfo)
        track.run()

        ### step 6: head detection

        ### step 7: writing data

        pprint(track.outdict)
        write_yaml(track.outdict_file, track.outdict)
    except:
        print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])
        pprint(track.outdict)
        write_yaml(track.outdict_file, track.outdict)


if __name__ == '__main__':
    main()
