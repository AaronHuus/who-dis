# Adapted from
# https://www.pyimagesearch.com/2016/01/04/unifying-picamera-and-cv2-videocapture-into-a-single-class-with-opencv/
from camera.FileVideoStream import FileVideoStream
from camera.WebcamVideoStream import WebcamVideoStream


class VideoStream:
    def __init__(self, src=0, use_pi_camera=False, resolution=(640, 480), framerate=32):

        video_width = resolution[0] or 640
        video_height = resolution[1] or 480
        # check to see if the picamera module should be used
        if use_pi_camera:
            # only import the picamera packages unless we are explicity told to do so -- this helps remove the
            # requirement of `picamera[array]` from desktops or laptops that still want to use the `imutils` package
            from camera.PiVideoStream import PiVideoStream

            # initialize the picamera stream and allow the camera sensor to warmup
            self.stream = PiVideoStream(resolution=(video_width, video_height), framerate=framerate)

        # otherwise, we are using OpenCV so initialize the webcam stream
        elif src != 0 and not src.startswith('http'):
            self.stream = FileVideoStream(src=src)
        else:
            self.stream = WebcamVideoStream(src=src, resolution=(video_width, video_height))

    def start(self):
        # start the threaded video stream
        return self.stream.start()

    def update(self):
        # grab the next frame from the stream
        self.stream.update()

    def read(self):
        # return the current frame
        return self.stream.read()

    def release(self):
        # stop the thread and release any resources
        self.stream.release()
