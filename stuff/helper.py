import os
import numpy as np
import datetime
from Xlib import display
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

from stuff.speech_synthesis import SpeechSynthesizer

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

    def draw(self, image_np, boxes, classes, scores):
        if not self._enabled:
          return

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

    def show(self, image_np):
        if not self._enabled:
          return True

        cv2.imshow('Visualizer', image_np) # alternatively as 2nd param: cv2.resize(image_np, (800, 600)))
        if not self._windowPlaced:
          cv2.moveWindow('Visualizer', (int)((self._screen.width-image_np.shape[1])/2), (int)((self._screen.height-image_np.shape[0])/2))
          self._windowPlaced = True
        if cv2.waitKey(1) & 0xFF == ord('q'):
          return False
        return True

    def cleanup(self):
        cv2.destroyAllWindows()


class Processor:
    def __init__(self):
        self._speech = SpeechSynthesizer()

    def process(self, boxes, scores, classes, num, image_shape):

        # TODO: There is the chance of overlapping detections, i.e., a caw and a dog are recognized either with exactly or very similar bounding boxes => filter those?

        obj = []

        print('*****')
        for i in range(boxes.shape[0]):
          if scores[i] > 0.5:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
            (left, right, top, bottom) = (int(xmin * image_shape[1]), int(xmax * image_shape[1]), int(ymin * image_shape[0]), int(ymax * image_shape[0]))
            obj.append([class_name, int(100*scores[i]), left, top, right, bottom])
            display_str = '{}: {}% at image coordinates (({}, {}) to ({}, {}))'.format(class_name, int(100*scores[i]), left, top, right, bottom)
            print(display_str)

        def getIndefiniteArticle(word):
            """Simplified way of choosing an or a for the following word; of course, there are many exceptions and not the letter but the sound (vowel vs. consonant) is important.
            But hey, for the COCO dataset there should not be any exceptions!
            See also https://www.englishclub.com/pronunciation/a-an.htm
            """
            return 'an' if word[:1].lower() in 'aeiou' else 'a'

        if(len(obj) > 0):
            self._speech.request("I am " + str(obj[0][1]) + "% certain I see " + getIndefiniteArticle(obj[0][0]) + " " + obj[0][0])

    def cleanup(self):
        pass
