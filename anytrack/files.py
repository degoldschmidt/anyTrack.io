import os
import os.path as op

def get_videos(input, output, video=None):
    if 'videos' in output:
        videos = [_file for _file in output['videos']]
        dir = output['directory']
    else:
        if video is None:
            dir = input
            videos = [op.join(input, _file) for _file in os.listdir(input) if _file.endswith('avi')]
        else:
            dir = op.dirname(input)
            videos = [video]
        #videos = [op.join(input, _file) for _file in checkbox('videos', videos, msg='Select videos for tracking:')]
    return videos, dir
