from __future__ import print_function, unicode_literals
import os, sys
import os.path as op
import cv2
import argparse
from PyInquirer import style_from_dict, Token, prompt, Separator
from pprint import pprint
import numpy as np
import pandas as pd
import io, yaml
from tqdm import tqdm

def dist(pos1, pos2):
    dx, dy = (pos1[0] - pos2[0]), (pos1[1] - pos2[1])
    return np.sqrt(dx * dx + dy * dy)

def read_yaml(_file):
    """ Returns a dict of a YAML-readable file '_file'. Returns None, if file is empty. """
    with open(_file, 'r') as stream:
        out = yaml.load(stream)
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
            (170, 88, 3)]

def checkbox(name, options, msg=None):
    style = style_from_dict({
    Token.Separator: '#27ae60',
    Token.QuestionMark: '#00d700 bold',
    Token.Selected: '#27ae60',  # default
    Token.Pointer: '#00d700 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#00d700 bold',
    Token.Question: '',
    })

    choices = [Separator('= {} ='.format(name))]
    for option in options:
        choices.append({'name': option, 'checked': True})

    if msg is None:
        msg = 'Select {}'.format(name)

    questions = [
        {
            'type': 'checkbox',
            'message': msg,
            'name': name,
            'choices': choices,
            'validate': lambda answer: 'You must choose at least one option.' \
                if len(answer) == 0 else True
        }
    ]

    return prompt(questions, style=style)[name]

def get_background(video, num_frames=30, how='uniform', offset=0, show_all=False):
    print('Start modelling background...', end='', flush=True)
    h, w = video.height, video.width
    bgframe = np.zeros((h, w), dtype=np.float32)
    if how=='uniform':
        choices = np.random.choice(video.len-1, num_frames)
    for i in range(num_frames):
        if how=='uniform':
            frameno = choices[i] + offset
        else:
            frameno = i + offset
        frame = video.get_frame(frameno)
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
        choices = np.random.choice(video.len-1, num_frames)
    for i in range(num_frames):
        if how=='uniform':
            frameno = choices[i] + offset
        else:
            frameno = i + offset
        frame = video.get_frame(frameno)
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
    print('[DONE]')
    return newbg

def get_videos(input):
    if op.isfile(input):
        dir = op.dirname(input)
        videos = [input]
    elif op.isdir(input):
        dir = input
        videos = [op.join(input, _file) for _file in os.listdir(input) if _file.endswith('avi')]
    else:
        raise FileNotFoundError
    return videos, dir

def load_trajectory(_file):
    df = pd.read_csv(_file)
    size = df.shape[0]
    trajectory = Trajectory(size)
    for i,col in enumerate(trajectory.columns):
        trajectory.data[:,i] = np.array(df[col])
    return trajectory

class Trajectory(object):
    def __init__(self, size):
        self.data = np.zeros((size,6))
        self.data[:] = np.nan
        self.columns = ['body_x', 'body_y', 'head_x', 'head_y', 'major', 'minor']
        self.x = self.data[:,0]
        self.y = self.data[:,1]
        self.hx = self.data[:,2]
        self.hy = self.data[:,3]
        self.major = self.data[:,4]
        self.minor = self.data[:,5]
        self.ohx, self.ohy = None, None
        self.i = 0

    def append(self, x, y, hx, hy, ma, mi):
        self.data[self.i, 0] = x
        self.data[self.i, 1] = y
        self.data[self.i, 2] = hx
        self.data[self.i, 3] = hy
        self.data[self.i, 4] = ma
        self.data[self.i, 5] = mi
        self.ohx, self.ohy = hx, hy
        self.i += 1

    def get_last_valid(self):
        return self.ohx, self.ohy

class Video(object):
    def __init__(self, __file):
        self.cap = cv2.VideoCapture(__file)
        self.file = __file
        self.base = op.basename(__file).split('.')[0]
        if self.cap is not None:
            self.len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.aratio = self.width/self.height

    def get_frame(self, frameno=-1):
        if frameno != -1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameno)
        ret, frame = self.cap.read() # read next frame, get next return code
        if ret:
            return frame
        else:
            return None

    def restart(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

class Tracking(object):
    def __init__(self, __files, output=None):
        if 'videos' in output:
            self.videos = [Video(_file) for _file in output['videos']]
        else:
            self.videos = [Video(_file) for _file in checkbox('videos', __files, msg='Select videos for tracking:')]
        self.outdict=output

    def get_statistics(self, video):
        no_cnts = []
        areas = []
        print('Detect contour statistics...', end='', flush=True)
        stats_frames = 20
        for frameno in tqdm(range(stats_frames)):
            choices = np.random.choice(video.len-1, stats_frames)
            frame = video.get_frame(choices[frameno])
            if thresholding == 'dark':
                bg = self.background.astype(np.float64)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                difference =  bg - gray_frame
            __, subtr = cv2.threshold(difference, threshold_level, 255, cv2.THRESH_BINARY)
            im2, contours, hierarchy = cv2.findContours(subtr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            no_cnts.append(len(contours))
            areas.append(np.mean([cv2.contourArea(cnt) for cnt in contours]))
        print('[DONE]')
        print('Detected {} contour(s) with mean area {:.1f} +- {:.1f}'.format(int(np.round(np.mean(no_cnts))), np.mean(areas), np.std(areas)))
        return int(np.round(np.mean(no_cnts))), np.mean(areas), np.std(areas)

    def run(self, threshold_level=30, thresholding='dark'):
        if 'background_files' not in self.outdict:
            self.outdict['background_files'] = {}
        for vid in self.videos:
            print('Track video: {}'.format(vid.file))
            if vid.file not in self.outdict['background_files']:
                self.outdict['background_files'][vid.file] = op.join(self.outdict['directory'], 'output_anytrack', '{}_bg.png'.format(op.basename(vid.file).split('.')[0]))
            if op.isfile(self.outdict['background_files'][vid.file]):
                print('Loading background file...', end='', flush=True)
                self.background = cv2.imread(self.outdict['background_files'][vid.file])[:,:,0]
                print('[DONE]')
            else:
                self.background = get_background(vid)
                cv2.imwrite(self.outdict['background_files'][vid.file], self.background)
            vid.restart()
            print('Start background subtraction')
            if 'n_contours' not in self.outdict:
                n = int(input('How many contours? '))
                self.outdict['n_contours'] = n
            else:
                n = self.outdict['n_contours']
            if 'countours_min_area' not in self.outdict:
                min_area = int(input('Minimal area? '))
                self.outdict['countours_min_area'] = min_area
            else:
                min_area = self.outdict['countours_min_area']
            tracks = [Trajectory(vid.len) for i in range(n)]
            print(len(tracks))
            vid.restart()
            for frameno in tqdm(range(vid.len)):
                frame = vid.get_frame()
                if thresholding == 'dark':
                    bg = self.background.astype(np.float64)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                    difference =  bg - gray_frame
                __, subtr = cv2.threshold(difference, threshold_level, 255, cv2.THRESH_BINARY)
                im2, contours, hierarchy = cv2.findContours(subtr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                output = frame.copy()
                contours = [cnt for cnt in contours if cv2.contourArea(cnt)>min_area]
                for j,cnt in enumerate(contours):
                    area = cv2.contourArea(cnt)
                    if area > 5:
                        ellipse = cv2.fitEllipse(cnt)
                        (x, y),(MA,ma),angle = ellipse
                        a = np.radians(angle) + np.pi/2
                        ax,ay = x+0.4*ma*np.cos(a), y+0.4*ma*np.sin(a)
                        apx = output[int(ay), int(ax),0]
                        bx,by = x-0.4*ma*np.cos(a), y-0.4*ma*np.sin(a)
                        bpx = output[int(by), int(bx),0]
                        old = (tracks[0].ohx, tracks[0].ohy)
                        if old[0] is None:
                            if apx < bpx:
                                hx, hy = ax, ay
                            else:
                                hx, hy = bx, by
                        elif dist(old, (ax,ay)) < dist(old, (bx,by)):
                            hx, hy = ax, ay
                        else:
                            hx, hy = bx, by
                        tracks[0].append(x,y,hx,hy,ma,MA)
                        dr = int(area * 0.025)
                        if j < len(colors):
                            #cv2.drawContours(output, [cnt], 0, colors[j], -1)
                            pts = tracks[0].data[:frameno+1,2:4].reshape((-1,1,2)).astype(np.int32)
                            #cv2.circle(output, (int(x),int(y)), 3, colors[j], 1)
                            cv2.circle(output, (int(hx),int(hy)), 5, (255,0,255), 1)
                            cv2.polylines(output,[pts],False,(255,0,255))
                            #cv2.putText(output, '{}'.format(apx), (int(ax)+dr, int(ay)+dr), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[j+1], 1, cv2.LINE_AA)
                            #cv2.circle(output, (int(bx),int(by)), 5, colors[j+2], 1)
                            #cv2.putText(output, '{}'.format(bpx), (int(bx)+dr, int(by)+dr), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[j+2], 1, cv2.LINE_AA)

                            #cv2.putText(output, '{} ({}, {}), area = {}'.format(j, int(x), int(y), int(area)), (int(x)+dr, int(y)+dr), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors[j], 2, cv2.LINE_AA)
                        else:
                            cv2.drawContours(output, [cnt], 0, (255,255,255), -1)
                cv2.imshow("Tracking", cv2.resize(output, (1400,1400)))
                cv2.waitKey(1) # time to wait between frames, in mSec
            for i, tr in enumerate(tracks):
                df = pd.DataFrame( { key: tr.data[:,i] for i, key in enumerate(tr.columns)})
                df.to_csv(op.join(self.outdict['directory'], 'output_anytrack', '{}_trajectory_{}.csv'.format(vid.base, i)))
            print('[DONE]')
        return self.outdict


def main():
    parser = argparse.ArgumentParser()
    # add required and optional arguments
    parser.add_argument('-i', dest='input', action='store',
                        help='input file(s)/directory')
    #parser.add_argument('--help', action='store_true')
    # namespace of arguments
    args = parser.parse_args()

    input = args.input
    videos, basedir = get_videos(input)
    outfolder = op.join(basedir, 'output_anytrack')
    os.makedirs(outfolder, exist_ok=True)
    outdict_file = op.join(outfolder, 'outdict.yaml')
    if op.isfile(outdict_file):
        outdict = read_yaml(outdict_file)
    else:
        outdict = {}
        outdict['directory'] = basedir
    track = Tracking(videos, output=outdict)
    outdict = track.run()
    write_yaml(outdict_file, outdict)

if __name__ == '__main__':
    main()
