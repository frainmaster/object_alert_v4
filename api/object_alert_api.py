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

            # workbook = xl.Workbook(output_path + vid_name + '_counter_oaa.xlsx')
            # worksheet = workbook.add_worksheet()
            # bold = workbook.add_format({'bold': True})

            # # write table header
            # worksheet.write('A1', 'TIME (S)', bold)
            # worksheet.write('B1', 'COUNT', bold)
            # worksheet.write('C1', 'CUMULATIVE', bold)

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

                # counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array(
                #     cap.get(1),
                #     cropped_frame,
                #     1,
                #     0,
                #     np.squeeze(boxes),
                #     np.squeeze(classes).astype(np.int32),
                #     np.squeeze(scores),
                #     category_index,
                #     use_normalized_coordinates=True,
                #     min_score_thresh=.5,
                #     line_thickness=4)

                # # when the vehicle passed over line and counted, make the color of ROI line green
                # if counter == 1:
                #     cv2.line(frame, (roi, 0), (roi, height), (0, 0xFF, 0), 1)
                # else:
                #     cv2.line(frame, (roi, 0), (roi, height), (0, 0, 0xFF), 1)

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

                # # when objects are detected, an alert will appear
                # # if counter > 0:
                # if counting_mode != '':
                #     cv2.putText(
                #         frame,
                #         'CAUTION',
                #         (10, 35),
                #         font,
                #         .8,
                #         (0, 0, 0xFF),
                #         2,
                #         cv2.LINE_AA)
                #     # if frame_counter % fps == 0:
                #     #     total_passed_object_per_interval += 1

                cv2.putText(
                        frame,
                        'Object count: ' + str(total_passed_object),
                        (10, 35),
                        font,
                        0.6,
                        (0,255,255),
                        2,
                        cv2.FONT_HERSHEY_SIMPLEX
                    )

                # to perform counting based on boundary box
                # if counting_mode != '':
                #     # cv2.putText(
                #     #     frame,
                #     #     'CAUTION',
                #     #     (10, 35),
                #     #     font,
                #     #     .8,
                #     #     (0, 0, 0xFF),
                #     #     2,
                #     #     cv2.LINE_AA)
                #     if frame_counter % fps == 0:
                #         total_passed_object_per_interval += 1

                output_video.write(frame)
                print ('writing frame ' + str(frame_counter) + '/' + str(total_frame))

                # # writing to excel file
                # if frame_counter % (interval_duration*fps) == 0:
                #     current_row = frame_counter//(interval_duration*fps)
                #     worksheet.write(current_row, 0, frame_counter//fps)
                #     worksheet.write(current_row, 1, total_passed_object_per_interval)
                #     if current_row == 1:
                #         worksheet.write(current_row, 2, '=B2')
                #     else:
                #         worksheet.write(current_row, 2, '=B' + str(current_row+1) + '+C' + str(current_row)) # =B(x+1)+C(x)
                #     total_passed_object_per_interval = 0

                # if frame_counter == total_frame:
                #     final_col = current_row

                frame_counter += 1

            # # crate the graph consisting of bar and line chart
            # bar_chart = workbook.add_chart({'type':'column'})
            # bar_chart.add_series({
            #     'name':'=Sheet1!B1',
            #     'categories':'=Sheet1!A2:A' + str(final_col+1),
            #     'values':'=Sheet1!B2:B' + str(final_col+1)
            #     })
            # line_chart = workbook.add_chart({'type':'line'})
            # line_chart.add_series({
            #     'name':'=Sheet1!C1',
            #     'categories':'=Sheet1!A2:A' + str(final_col+1),
            #     'values':'=Sheet1!C2:C' + str(final_col+1)
            #     })
            # bar_chart.combine(line_chart)

            # bar_chart.set_title({'name':'No of Alerts'})
            # bar_chart.set_x_axis({'name':'=Sheet1!A1'})
            # bar_chart.set_y_axis({'name':'Alert count'})
            # worksheet.insert_chart('F2', bar_chart)

            # # mode, median and mean of data (count)
            # worksheet.write('A' + str(final_col+3), 'MODE', bold)
            # worksheet.write('A' + str(final_col+4), 'MEDIAN', bold)
            # worksheet.write('A' + str(final_col+5), 'MEAN', bold)
            # worksheet.write('A' + str(final_col+6), 'SD', bold)
            # worksheet.write('B' + str(final_col+3), '=MODE(B2:B' + str(final_col+1) + ')')
            # worksheet.write('B' + str(final_col+4), '=MEDIAN(B2:B' + str(final_col+1) + ')')
            # worksheet.write('B' + str(final_col+5), '=AVERAGE(B2:B' + str(final_col+1) + ')')
            # worksheet.write('B' + str(final_col+6), '=STDEV(B2:B' + str(final_col+1) + ')')
            # worksheet.write('C' + str(final_col+3), '=MODE(C2:C' + str(final_col+1) + ')')
            # worksheet.write('C' + str(final_col+4), '=MEDIAN(C2:C' + str(final_col+1) + ')')
            # worksheet.write('C' + str(final_col+5), '=AVERAGE(C2:C' + str(final_col+1) + ')')
            # worksheet.write('C' + str(final_col+6), '=STDEV(C2:C' + str(final_col+1) + ')')

            # workbook.close()

            cap.release()
            cv2.destroyAllWindows()
