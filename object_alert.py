#!/usr/bin/python3.7

"""
----------------------------------------------
--- Author         : Irfan Ahmad
--- Mail           : irfanibnuahmad@gmail.com
--- Date           : 18th Oct 2020
----------------------------------------------
"""

from os import listdir

input_video_path = './input/'

data = [
    {"input_video": "Video_Danger_Alert.mp4", "start_point": (224, 52), "end_point": (416, 281)},
    {"input_video": "level_crossing_bahau.mp4", "start_point": (223, 40), "end_point": (445, 611)},
    {"input_video": "level crossing bahau.mp4", "start_point": (223, 40), "end_point": (445, 611)},
    {"input_video": "train_video.mp4", "start_point": (103, 205), "end_point": (335, 588)},
    {"input_video": "lrt_bandaraya_peak.mp4.mp4", "start_point": (146, 130), "end_point": (401, 337)}
]
list = [k for k in listdir(input_video_path) if k != "res"]

# select video here
print("select video:\n")
for i, j in enumerate(list):
    print(i+1, j)
selected_vid = list[int(input())-1]
print(selected_vid)

input_video = start_point = end_point = 0

for i in data:
    if i["input_video"] == selected_vid:
        input_video = selected_vid
        start_point, end_point = i["start_point"], i["end_point"]

if input_video == 0:
    raise IOError("data not found. please setup data")

roi = (start_point[0]+end_point[0])//2

# Object detection imports
from utils import backbone
from api import object_alert_api as oa_api

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03', 'mscoco_label_map.pbtxt')  # 26 ms
# detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')  # 30 ms
# detection_graph, category_index = backbone.set_model('faster_rcnn_inception_v2_coco_2018_01_28', 'mscoco_label_map.pbtxt')  # 58 ms

oa_api.object_alert(input_video_path + input_video, detection_graph, category_index, start_point, end_point, roi)