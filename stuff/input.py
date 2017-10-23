import os
import ctypes
import numpy as np
from Xlib import display, X
from PIL import Image
import cv2

class ScreenInput:
    # Natively captures the screen using Xlib and our pre-compiled grab_screen library
    # see also https://stackoverflow.com/a/16141058/860756
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.width = endX-startX
        self.height = endY-startY

        self._grab = ctypes.CDLL(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'grab_screen.so')
        self._size = ctypes.c_ubyte * self.width * self.height * 3

    def isActive(self):
        return True

    def getImage(self):
        self._grab.getScreen.argtypes = []
        result = (self._size)()
        self._grab.getScreen(self.startX,self.startY, self.width, self.height, result)
        image = Image.frombuffer('RGB', (self.width, self.height), result, 'raw', 'RGB', 0, 1)
        image_np = np.array(image);
        return True, image_np

    def cleanup(self):
        pass

class ScreenPyInput:
    # Capture the screen using Xlib and Python-only (slower)
    def __init__(self, startX, startY, endX, endY):
        self.startX = startX
        self.startY = startY
        self.width = endX-startX
        self.height = endY-startY

        self.root = display.Display().screen().root
        self.reso = self.root.get_geometry()

    def isActive(self):
        return True

    def getImage(self):
        raw = self.root.get_image(self.startX, self.startY, self.width, self.height, X.ZPixmap, 0xffffffff)
        image = Image.frombytes("RGB", (self.width, self.height), raw.data, "raw", "RGBX")
        image_np = np.array(image);
        return True, image_np

    def cleanup(self):
        pass

class VideoInput:
    # Capture video (either via device (e.g. camera) or video files) using OpenCV
    def __init__(self, input):
        self.cap = cv2.VideoCapture(input)

    def isActive(self):
        return self.cap.isOpened()

    def getImage(self):
        return self.cap.read()

    def cleanup(self):
        self.cap.release()
