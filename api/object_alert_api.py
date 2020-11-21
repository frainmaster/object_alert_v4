import tensorflow.compat.v1 as tf
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import xlsxwriter as xl
import subprocess
import re

tf.disable_v2_behavior()


def get_orientation(input_video):
    orientation = 0

    stdout = stderr = p = ""

    try:
        p = subprocess.Popen(
            ["ffmpeg", "-i", input_video],
            stderr = subprocess.PIPE,
            close_fds=True
        )
        stdout, stderr = p.communicate()
    except FileNotFoundError:
        print('File ffmpeg.exe not found')
        return 0

    reo_rotation = re.compile('rotate\s+:\s(?P<rotation>.*)')
    match_rotation = reo_rotation.search(str(stderr))
    try:
        rotation = match_rotation.groups()[0]
        orientation = int(str(rotation)[:str(rotation).find('\\')])
    except AttributeError:
        print('video is in the right orientation')
    return orientation


def object_alert(input_video, detection_graph, category_index, start_point, end_point, roi):

    api_suffix = '_oaa'

    total_passed_object = 0
    output_path = 'output/'
    cap = cv2.VideoCapture(input_video)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid_name = input_video[input_video.rfind('/')+1: input_video.rfind('.')]

    orientation = get_orientation(input_video)
    print('orientation: ' + str(orientation))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if orientation in [90, 270]:
        height, width = width, height
    output_video = cv2.VideoWriter(output_path + vid_name + api_suffix + '.mp4', fourcc, fps, (width, height))

    print('height: ' + str(height))
    print('width: ' + str(width))
    print('frame count: ' + str(total_frame))
    print('fps: ' + str(fps))
    print('video name: ' + vid_name)

    start_warning = 0

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # variables to work with excel
            frame_counter = 1
            total_passed_object_per_interval = 0
            final_col = 0
            current_row = 0
            interval_duration = 5

            # for all the frames that are extracted from input video
            while(cap.isOpened()):
                ret, frame = cap.read()
                if orientation == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif orientation == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if not ret:
                    print("end of the video file...")
                    break

                # draw rectangle; blue, line thickness=1
                (w1, h1), (w2, h2) = start_point, end_point
                cv2.rectangle(frame, start_point, end_point, (255, 0, 0), 1)
                cropped_frame = frame[h1:h2, w1:w2]

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(cropped_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # insert information text to video frame
                font = cv2.FONT_HERSHEY_SIMPLEX

                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(
                    cap.get(1),
                    cropped_frame,
                    1,
                    0,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    x_reference = roi,
                    deviation = 1,
                    use_normalized_coordinates=True,
                    min_score_thresh=.6,
                    line_thickness=2)

                total_passed_object += counter
                total_passed_object_per_interval += counter

                for a, b in enumerate(boxes[0]):
                    # if classes[0][a] in [9]: # tis a boat
                    # if classes[0][a] in [1, 2, 3, 4, 6, 7, 8]:
                    #     print(classes[0][a], counting_mode)
                        if scores[0][a] > .6:
                            mid_x = (boxes[0][a][3] + boxes[0][a][1])/2
                            mid_y = (boxes[0][a][2] + boxes[0][a][0])/2
                            apx_distance = round((1-(boxes[0][a][3] - boxes[0][a][1]))**4, 1)
                            # cv2.putText(
                            #     cropped_frame,
                            #     # frame,
                            #     '{}'.format(apx_distance),
                            #     (int(mid_x*width), int(mid_y*height)),
                            #     # (int(mid_x), int(mid_y)),
                            #     font,
                            #     .7,
                            #     (255, 255, 255),
                            #     2
                            #     )
                            if apx_distance <= .6:
                                if mid_x > .1 and mid_x < .9:
                                    cv2.putText(
                                        frame,
                                        'CAUTION',
                                        # (10, 70),
                                        ((w1+w2)//2-40, (h1+h2)//2+5),
                                        font,
                                        .7,
                                        (0, 0, 255),
                                        2
                                    )
                                    start_warning = frame_counter + fps * 5

                if frame_counter < start_warning:
                    cv2.putText(
                        frame,
                        'CAUTION',
                        # (10, 70),
                        ((w1+w2)//2-40, (h1+h2)//2+5),
                        font,
                        .7,
                        (0, 0, 255),
                        2
                    )

#                 cv2.putText(
#                         frame,
#                         'Object count: ' + str(total_passed_object),
#                         (10, 35),
#                         font,
#                         0.6,
#                         (0,255,255),
#                         2,
#                         cv2.FONT_HERSHEY_SIMPLEX
#                     )

                output_video.write(frame)
                print ('writing frame ' + str(frame_counter) + '/' + str(total_frame))

                frame_counter += 1

            cap.release()
            cv2.destroyAllWindows()
