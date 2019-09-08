import argparse
from pprint import pprint
import os.path as op

from anytracker import Anytracker

def main(input):
    ### create AnyTrack Tracking object
    track = Anytracker(input=input, output='output_anytrack')
    ### step 1: batch run ROIfinder to get arenas (automated/supervised/manual) recommended to use 'supervised'
    track.roi_select(method='supervised')

    ### step 2: get image statistics (average signal over time, )
    avg_pxs = track.image_stats(skip=30)

    ### step 3: background modelling (DONE)
    track.model_bg(baselines=avg_pxs)

    ### step 4: infer contour statistics (DONE)
    track.infer()

    ### step 5: background subtraction & contour matching & centroid fit & identity (DONE, frame -> contours -> centroid+pixelinfo)
    flytracks = track.run(nframes=1000, show=0)

    ### step 6: head detection
    flytracks = track.detect_head(flytracks)

    ### step 7: writing data
    track.outdict['trajectory_files'] = {}
    for video in track.videos:
        track.outdict['trajectory_files'][video] = []
        for i,fly in enumerate(flytracks[video]):
            track.outdict['trajectory_files'][video].append(op.join(track.outdict['folders']['trajectories'], '{}_fly{}.csv'.format(op.basename(video).split('.')[0],i)))
            fly.save(op.join(track.outdict['folders']['trajectories'], '{}_fly{}.csv'.format(op.basename(video).split('.')[0],i)))

    pprint(track.outdict)
    write_yaml(track.outdict_file, track.outdict)

if __name__ == '__main__':
    ### arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', action='store',
                        help='input file(s)/directory')
    args = parser.parse_args()
    input = args.input
    main(input)
