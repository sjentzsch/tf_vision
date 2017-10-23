import os
import numpy as np
import datetime
from Xlib import display
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

# Loading label map (mapping indices to category names, e.g. 5 -> airplane)
NUM_CLASSES = 90
PATH_TO_LABELS = os.path.join('../tensorflow_models/research/object_detection/data', 'mscoco_label_map.pbtxt')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class FPS:
    def __init__(self, interval):
        self._glob_start = None
        self._glob_end = None
        self._glob_numFrames = 0
        self._local_start = None
        self._local_numFrames = 0
        self._interval = interval

    def start(self):
        self._glob_start = datetime.datetime.now()
        self._local_start = self._glob_start
        return self

    def stop(self):
        self._glob_end = datetime.datetime.now()

    def update(self):
        curr_time = datetime.datetime.now()
        curr_local_elapsed = (curr_time - self._local_start).total_seconds()
        self._glob_numFrames += 1
        self._local_numFrames += 1
        if curr_local_elapsed > self._interval:
          print("FPS: ", self._local_numFrames / curr_local_elapsed)
          self._local_numFrames = 0
          self._local_start = curr_time

    def elapsed(self):
        return (self._glob_end - self._glob_start).total_seconds()

    def fps(self):
        return self._glob_numFrames / self.elapsed()


class Visualizer:
    def __init__(self, enabled):
        self._enabled = enabled
        self._windowPlaced = False
        self._screen = display.Display().screen().root.get_geometry()

    def show(self, image_np, boxes, classes, scores):
        if not self._enabled:
          return True

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow('Visualizer', image_np) # alternatively as 2nd param: cv2.resize(image_np, (800, 600)))
        if not self._windowPlaced:
          cv2.moveWindow('Visualizer', (int)((self._screen.width-image_np.shape[1])/2), (int)((self._screen.height-image_np.shape[0])/2))
          self._windowPlaced = True
        if cv2.waitKey(1) & 0xFF == ord('q'):
          return False
        return True

    def cleanup(self):
        cv2.destroyAllWindows()
