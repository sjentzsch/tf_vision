import numpy as np
from Xlib import display, X
from PIL import Image
import cv2


class ScreenInput:
    def __init__(self, startX, startY, endX, endY):
        self.root = display.Display().screen().root
        self.reso = self.root.get_geometry()

        self.startX = startX
        self.startY = startY
        self.width = endX-startX
        self.height = endY-startY

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
    def __init__(self, input):
        self.cap = cv2.VideoCapture(input)

    def isActive(self):
        return self.cap.isOpened()

    def getImage(self):
        return self.cap.read()

    def cleanup(self):
        self.cap.release()
