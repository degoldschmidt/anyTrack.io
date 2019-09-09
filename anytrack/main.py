import argparse
from pprint import pprint
import os.path as op

from anytrack.anytracker import Anytracker
from anytrack.yaml_helpers import read_yaml, write_yaml

def main():
    ### arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', action='store',
                        help='input file(s)/directory')
    parser.add_argument('-n', dest='num_contours', action='store', type=int,
                        help='fix number of contours')
    args = parser.parse_args()
    input = args.input
    num_contours = args.num_contours
    ### create AnyTrack Tracking object
    track = Anytracker(input=input, output='output_anytrack')
    ### step 1: batch run ROIfinder to get arenas (automated/supervised/manual) recommended to use 'supervised'
    track.roi_select(method='supervised')
    write_yaml(track.outdict_file, track.outdict)


    ### step 2: get image statistics (average signal over time, )
    avg_pxs = track.image_stats(skip=30)
    write_yaml(track.outdict_file, track.outdict)

    ### step 3: background modelling (DONE)
    track.model_bg(baselines=avg_pxs)
    write_yaml(track.outdict_file, track.outdict)

    ### step 4: infer contour statistics (DONE)
    track.infer(num_contours=num_contours)
    write_yaml(track.outdict_file, track.outdict)

    ### step 5: background subtraction & contour matching & centroid fit & identity (DONE, frame -> contours -> centroid+pixelinfo) &&&& head detection & wriring data
    flytracks = track.run(show=0)

    pprint(track.outdict)
    write_yaml(track.outdict_file, track.outdict)

if __name__ == '__main__':
    main()
