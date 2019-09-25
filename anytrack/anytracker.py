import cv2
import numpy as np
import os
import os.path as op
import sys
from tqdm import tqdm

from anytrack.background import get_background
from anytrack.contours import get_contours, get_contour_stats
from anytrack.files import get_videos
from anytrack.helpers import rle, nan_helper, interpolate, smooth, px, dist, in_roi
from anytrack.roiselect import arenaROIselector
from anytrack.trajectory import Trajectory
from anytrack.video import VideoCapture
from anytrack.yaml_helpers import read_yaml, write_yaml

def PrintException():
    import linecache
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))

def run_bg_subtraction(video, background=None, nframes=0, threshold_level=10, thresholding='dark', show=1, n_contours=4, min_size=150, max_size=250, rois=None):
    cap = VideoCapture(video,0)
    flag=False
    if nframes>0:
        nframes = nframes
    else:
        nframes = cap.len
    flytracks = [Trajectory(nframes) for i in range(n_contours)]
    for frameno in tqdm(range(nframes)):
        _,frame = cap.read()
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
        contour_mapping = [[in_roi(cv2.fitEllipse(cnt)[0],roi) for roi in rois].index(True) for cnt in contours] ### list of roi IDs for each contour
        if len(contours) == n_contours:
            order = contour_mapping
            for j,cnt in zip(order,contours):
                ellipse = cv2.fitEllipse(cnt)
                (x,y),(ma,mi),a = ellipse

                ax, ay = x+0.4*mi*np.cos(np.radians(a)+np.pi/2), y+0.4*mi*np.sin(np.radians(a)+np.pi/2)
                ox, oy = x-0.4*mi*np.cos(np.radians(a)+np.pi/2), y-0.4*mi*np.sin(np.radians(a)+np.pi/2)
                if frameno>0:
                    oax, oay = flytracks[j].data[frameno-1,8], flytracks[j].data[frameno-1,9]
                    oox, ooy = flytracks[j].data[frameno-1,10], flytracks[j].data[frameno-1,11]
                    if dist((ax, ay), (oax, oay))+dist((ox, oy), (oox, ooy))  > dist((ox, oy), (oax, oay))+dist((ax, ay), (oox, ooy)):
                        ax,ay,ox,oy=ox,oy,ax,ay
                px_a, px_o = frame[px(ay),px(ax),0], frame[px(oy),px(ox),0]
                flytracks[j].append(i=frameno,x=x,y=y,ma=mi,mi=ma,angle=a,avg_px=np.nanmean(frame), ax=ax, ay=ay, ox=ox, oy=oy, apx=px_a, opx=px_o)
                if show > 0:
                    cv2.circle(output,(px(x),px(y)), 1, (255,0,255),1)
                    cv2.circle(output,(px(ax),px(ay)), 2, (255,0,0),1)
                    cv2.circle(output,(px(ox),px(oy)), 2, (0,0,255),1)
                    cv2.putText(output, '{}'.format(px_a), (px(ax)+5, px(ay)+5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 1, cv2.LINE_AA)
                    cv2.putText(output, '{}'.format(px_o), (px(ox)+5, px(oy)+5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,255), 1, cv2.LINE_AA)
                    pts = flytracks[j].data[:frameno,0:2].astype(np.int32)
                    pts = pts.reshape((-1,1,2))
                    cv2.polylines(output,[pts],False,(0,255,255))
                    cv2.putText(output, '{}'.format(j), (px(x)+10, px(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 1, cv2.LINE_AA)
            for id, arena in enumerate(rois):
                ### draw arenas
                x, y, r, o  = int(arena['x']), int(arena['y']), int(arena['radius']), int(arena['outer'])
                color = (255,255,255)
                if show > 0:
                    cv2.circle(output,(x, y),r,color,1)
                    cv2.circle(output,(x, y),int(1.1*o),color,1)
                    cv2.circle(output,(x, y),1,color,-1)
                    cv2.rectangle(output, (px(x-1.5*r),px(y-1.5*r)), (px(x+1.5*r), px(y+1.5*r)), color, 1)
                    cv2.putText(output, str(id), (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX , .5, color, 1, cv2.LINE_AA)
        if show>0 or flag:
            if frameno%show==0 or flag:
                x,y,r = flytracks[0].data[frameno,0], flytracks[0].data[frameno,1], 100
                cv2.imshow("Tracking", cv2.resize(output[int(y-r):int(y+r), int(x-r):int(x+r)], (700,700)))
                if flag:
                    cv2.waitKey(0) # time to wait between frames, in mSec
                    flag = False
                else:
                    cv2.waitKey(1) # time to wait between frames, in mSec
    return flytracks

class Anytracker(object):
    def __init__(self, input=None, output=None):
        video=None
        if op.isfile(input):
            video = input
            input = op.dirname(input)
        ### create output folder inside input folder
        outfolder = op.join(input, output)
        os.makedirs(outfolder, exist_ok=True)
        outdict_file = op.join(outfolder, 'outdict.yaml')
        ### config file for multiple runs
        if op.isfile(outdict_file):
            outdict = read_yaml(outdict_file)
        else:
            outdict = {}
        #if 'folders' not in outdict:
        outdict['directory'] = input
        outdict['folders'] = {}
        outdict['folders']['background'] = op.join(outfolder, 'bg')
        outdict['folders']['image_stats'] = op.join(outfolder, 'image_stats')
        outdict['folders']['trajectories'] = op.join(outfolder, 'trajectories')
        outdict['folders']['output'] = outfolder
        for f in outdict['folders']:
            os.makedirs(outdict['folders'][f], exist_ok=True)
        ### getting videos based on input
        self.videos, basedir = get_videos(input, outdict, video=video)
        outdict['videos'] = self.videos
        self.outdict=outdict
        self.outdict_file = outdict_file
        self.background = {}

    def infer(self, num_contours):
        print('Infer contour statistics...', flush=True)
        if 'number_contours' not in self.outdict:
            self.outdict['number_contours'] = {}
            self.outdict['min_size'] = {}
            self.outdict['max_size'] = {}
        for video in tqdm(self.videos):
            if video in self.outdict['number_contours']:
                numcnts, min_size, max_size = self.outdict['number_contours'][video], self.outdict['min_size'][video], self.outdict['max_size'][video]
            else:
                numcnts, min_size, max_size = get_contour_stats(video, self.background[video], force_num_contours=num_contours)
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
                self.background[video] = cv2.imread(self.outdict['background_files'][video])[:,:,0]
            else:
                if baselines is not None:
                    only_these = np.where(baselines[video]==0)[0]
                else:
                    only_these = None
                self.background[video] = get_background(video, frames=only_these)
                cv2.imwrite(self.outdict['background_files'][video], self.background[video])

    def overlay(self):
        for video in tqdm(self.videos):
            for _file in os.listdir(self.outdict['folders']['trajectories']):
                pass

    def run(self, nframes=0, threshold_level=10, thresholding='dark', use_threads=1, show=0):
        print('Run tracking...', flush=True)
        all_tracks = {}
        self.outdict['trajectory_files'] = {}
        for video in tqdm(self.videos):
            ### run tracking
            try:
                tracks = run_bg_subtraction(    video,
                                                nframes=nframes,
                                                background=self.background[video],
                                                threshold_level=threshold_level,
                                                thresholding=thresholding,
                                                n_contours=self.outdict['number_contours'][video],
                                                min_size=self.outdict['min_size'][video],
                                                max_size=self.outdict['max_size'][video],
                                                rois=self.outdict['ROIs'][video],
                                                show=show,)
                ### interpolate all signals
                for fly in tracks:
                    for i in range(fly.data.shape[1]):
                        if not all(np.isnan(fly.data[:,i])):
                            fly.data[:,i] = interpolate(fly.data[:,i])
                all_tracks[video] = tracks
            except:
                all_tracks[video] = None
                PrintException()

            for fly in tracks:
                windowlen=13
                apx = smooth(fly.data[:,12], windowlen)
                opx = smooth(fly.data[:,13], windowlen)
                diffs = opx - apx
                binary = np.sign(diffs)
                for rl, pos, state in zip(*rle(binary)):
                    if rl > 10: ### accept only valid switches longer than 10 frames
                        if state == 1:
                            fly.data[pos:pos+rl,2:4] = fly.data[pos:pos+rl,8:10]
                        elif state == -1:
                            fly.data[pos:pos+rl,2:4] = fly.data[pos:pos+rl,10:12]
                        else:
                            print(state, 'WEIRD')
                    else:
                        if state == 1:
                            fly.data[pos:pos+rl,2:4] = fly.data[pos:pos+rl,10:12]
                        elif state == -1:
                            fly.data[pos:pos+rl,2:4] = fly.data[pos:pos+rl,8:10]
                        else:
                            print(state, 'WEIRD')
                fly.data[:,6] = np.arctan2(fly.data[:,3] - fly.data[:,1], fly.data[:,2] - fly.data[:,0])

            ### save tracks
            self.outdict['trajectory_files'][video] = []
            for i,fly in enumerate(tracks):
                if fly is not None:
                    self.outdict['trajectory_files'][video].append(op.join(self.outdict['folders']['trajectories'], '{}_fly{}.csv'.format(op.basename(video).split('.')[0],i)))
                    fly.save(op.join(self.outdict['folders']['trajectories'], '{}_fly{}.csv'.format(op.basename(video).split('.')[0],i)))

        return all_tracks


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
