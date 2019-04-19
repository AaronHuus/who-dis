# Adapted from
# https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/
import time
from threading import Thread
import cv2


class FileVideoStream:
    def __init__(self, src):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        return self

    def read(self):
        # return the frame most recently read
        (self.grabbed, self.frame) = self.stream.read()
        return self.frame

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True
