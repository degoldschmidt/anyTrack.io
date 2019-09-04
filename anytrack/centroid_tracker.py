import cv2

class CentroidTracker(object):
    def __init__(self):
        self.centroids = []
        self.trajectory_data = None

    def update(self, threshold_img):
        im2, contours, hierarchy = cv2.findContours(threshold_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
