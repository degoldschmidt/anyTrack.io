import argparse
from test_anytrack import Tracking

def main(input):
    ### create AnyTrack Tracking object
    track = Tracking(input=input, output='output_anytrack')
    for 

if __name__ == '__main__':
    ### arguments parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='input', action='store',
                        help='input file(s)/directory')
    args = parser.parse_args()
    input = args.input
    main(input)
