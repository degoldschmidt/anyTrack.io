import numpy as np
import cv2

from anytrack.video import VideoCapture

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
    contours, hierarchy = cv2.findContours(subtr.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_contour_stats(video, background, force_num_contours=None, num_frames=50, how='uniform'):
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
    if force_num_contours is None:
        area_contours = np.zeros((num_frames, int(round(mean_number))))
    else:
        area_contours = np.zeros((num_frames, force_num_contours))
    for i in range(num_frames):
        if how=='uniform':
            frameno = choices[i]
        else:
            frameno = i
        frame = cap.get_frame(frameno)
        contours = get_contours(frame, background)
        areas = sorted([cv2.contourArea(cnt) for cnt in contours], key=lambda x: np.abs(x-mean_area))[:int(round(mean_number))]
        if force_num_contours is None:
            area_contours[i, :] = np.array(areas[:int(round(mean_number))])
        else:
            area_contours[i, :] = np.array(areas[:force_num_contours])
    if force_num_contours is None:
        return mean_number, np.amin(area_contours), np.amax(area_contours)
    else:
        return force_num_contours, np.amin(area_contours), np.amax(area_contours)
