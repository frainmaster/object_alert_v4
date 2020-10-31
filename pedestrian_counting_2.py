#!/usr/bin/python3.7

# Object detection imports
from utils import backbone
from api import cumul_object_counting_api_old as coc_api

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03', 'mscoco_label_map.pbtxt') # 26 ms
# detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt') # 30 ms
# detection_graph, category_index = backbone.set_model('faster_rcnn_inception_v2_coco_2018_01_28', 'mscoco_label_map.pbtxt') # 58 ms

input_video_path = './input/'

# # default values
inter = 3
# is_color_recognition_enabled = 0
# deviation = 1

##########################################################
##########################################################

input_video = 'level crossing bahau.mp4'
roi = .5		# roi line position; input value in between 0 to 1 to set the line by percentage of video size
roi_axis = 0	# axis of ROI line; 0=x, 1=y

##########################################################
##########################################################

coc_api.cumul_object_counting_roi_line(input_video_path + input_video, detection_graph, category_index, roi, roi_axis, interval=inter)
