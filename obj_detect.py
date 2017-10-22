import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from Xlib import display
import cv2
import yaml


from collections import defaultdict
from io import StringIO
#from PIL import Image

sys.path.append('../tensorflow_models/research')
sys.path.append('../tensorflow_models/research/slim')
sys.path.append('../tensorflow_models/research/object_detection')

from utils import label_map_util
from utils import visualization_utils as vis_util

from stuff.helper import FPS
from stuff.input import ScreenInput, VideoInput

# Load config values from config.obj_detect.sample.yml (as default values) updated by optional user-specific config.obj_detect.yml
## see also http://treyhunner.com/2016/02/how-to-merge-dictionaries-in-python/
cfg = yaml.load(open("config/config.obj_detect.sample.yml", 'r'))
if os.path.isfile("config/config.obj_detect.yml"):
  cfg_user = yaml.load(open("config/config.obj_detect.yml", 'r'))
  cfg.update(cfg_user)
#for section in cfg:
#  print(section, ":", cfg[section])

# Define input
screen = display.Display().screen().root.get_geometry()
if cfg['input_type'] == 'screen':
  input = ScreenInput(0, 0, int(screen.width/2), int(screen.height/2))
elif cfg['input_type'] == 'video':
  input = VideoInput(cfg['input_video'])
else:
  print('No valid input type given. Exit.')
  sys.exit()

# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.
# See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '../' + cfg['model_name'] + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('../tensorflow_models/research/object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model
MODEL_FILE = cfg['model_name'] + cfg['model_dl_file_format']
if not os.path.isfile(PATH_TO_CKPT):
  print('Model not found. We will download it now.')
  opener = urllib.request.URLopener()
  opener.retrieve(cfg['model_dl_base_path'] + MODEL_FILE, '../' + MODEL_FILE)
  tar_file = tarfile.open('../' + MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd() + '/../')
  os.remove('../' + MODEL_FILE)
else:
  print('Model found. Proceed.')

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# # Detection
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # TODO: Usually FPS calculation lives in a separate thread. As is now, the interval is a minimum value for each iteration.
    fps = FPS(cfg['fps_interval']).start()

    windowPlacedYet = False

    while(input.isActive()):
        ret, image_np = input.getImage()
        if not ret:
          print("No frames grabbed from input (anymore)! Exit.")
          break

#        image_np_bgr = np.array(ImageGrab.grab(bbox=(0,0,600,600))) # grab(bbox=(10,10,500,500)) or just grab()
#        image_np = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)

#    for image_path in TEST_IMAGE_PATHS:
#      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
#      image_np = load_image_into_numpy_array(image)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow('object detection', image_np) # alternatively as 2nd param: cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if not windowPlacedYet:
          cv2.moveWindow('object detection', (int)(screen.width/3), (int)(screen.height/3))
          windowPlacedYet = True

        fps.update()

fps.stop()
print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

input.cleanup()
cv2.destroyAllWindows()
