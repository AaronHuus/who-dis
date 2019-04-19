class VideoFrame:

    def __init__(self, frame=[]):
        self._frame = frame

################
# Properties
################
    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, value):
        self._frame = value
