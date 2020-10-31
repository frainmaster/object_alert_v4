#----------------------------------------------
#--- Author         : Irfan Ahmad
#--- Mail           : frainmaster@gmail.com
#--- Date           : 27th June 2020
#----------------------------------------------

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import numpy as np
from utils import visualization_utils as vis_util
import xlsxwriter as xl

# parameters:
	# input_video						name of input video
	# detection_graph					from backbone.set_model
	# category_index					from backbone.set_model
	# roi								line position, can be by percentage of video size
	# roi_axis							axis of ROI line; 0=x, 1=y

# optional parameters
	# interval=5						interval to write into excel file, in seconds
	# is_color_recognition_enabled=0	status of colour recognition; 0=disabled, 1=enabled
	# deviation=1						constant that represents the object counting area

import subprocess, re

def get_orientation(input_video):
	orientation = 0

	p = subprocess.Popen(
		["ffmpeg", "-i", input_video],
		stderr = subprocess.PIPE,
		close_fds=True
	)
	stdout, stderr = p.communicate()

	reo_rotation = re.compile('rotate\s+:\s(?P<rotation>.*)')
	match_rotation = reo_rotation.search(str(stderr))
	try:
		rotation = match_rotation.groups()[0]
		orientation = int(str(rotation)[:str(rotation).find('\\')])
	except AttributeError:
		print('video is in the right orientation')
	return orientation

def cumul_object_counting_roi_line(input_video, detection_graph, category_index, roi, roi_axis, interval=5, is_color_recognition_enabled=0, deviation=1):

	total_passed_object = 0
	# output path
	output_path = 'output/'
	# input video
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
	output_video = cv2.VideoWriter(output_path + vid_name + '.mp4', fourcc, fps, (width, height))

	print('height: ' + str(height))
	print('width: ' + str(width))
	print('frame count: ' + str(total_frame))
	print('fps: ' + str(fps))
	print('video name: ' + vid_name)

	# set roi by percentage of video size
	if roi > 0 and roi < 1:
		if roi_axis == 0:
			roi = int(width*roi)
		elif roi_axis == 1:
			roi = int(height*roi)

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

			# variables to work with excel
			frame_counter = 1
			total_passed_object_per_interval = 0
			final_col = 0
			current_row = 0

			workbook = xl.Workbook(output_path + vid_name + '_' + str(interval) + 's_interval.xlsx')
			worksheet = workbook.add_worksheet()
			bold = workbook.add_format({'bold': True})

			# write table header
			worksheet.write('A1', 'TIME (S)', bold)
			worksheet.write('B1', 'COUNT', bold)
			worksheet.write('C1', 'CUMULATIVE', bold)

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

				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(frame, axis=0)

				# Actual detection.
				(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})

				# insert information text to video frame
				font = cv2.FONT_HERSHEY_SIMPLEX

				# Visualization of the results of a detection
				if roi_axis == 0:	# if x axis
					counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(
						cap.get(1),
						frame,
						1,
						is_color_recognition_enabled,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						x_reference = roi,
						deviation = deviation,
						use_normalized_coordinates=True,
						line_thickness=4)

					# when the vehicle passed over line and counted, make the color of ROI line green
					if counter == 1:
						cv2.line(frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
					else:
						cv2.line(frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

				elif roi_axis == 1:	# if y axis
					counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(
						cap.get(1),
						frame,
						2,
						is_color_recognition_enabled,
						np.squeeze(boxes),
						np.squeeze(classes).astype(np.int32),
						np.squeeze(scores),
						category_index,
						y_reference = roi,
						deviation = deviation,
						use_normalized_coordinates=True,
						line_thickness=4)

					# when the vehicle passed over line and counted, make the color of ROI line green
					if counter == 1:
						cv2.line(frame, (0, roi), (width, roi), (0, 0xFF, 0), 5)
					else:
						cv2.line(frame, (0, roi), (width, roi), (0, 0, 0xFF), 5)

				total_passed_object += counter
				total_passed_object_per_interval += counter

				# insert information text to video frame
				cv2.putText(
					frame,
					'Detected objects: ' + str(total_passed_object),
					(10, 35),
					font,
					0.8,
					(0, 0xFF, 0xFF),
					2,
					font,
					)

				cv2.putText(
					frame,
					'ROI Line',
					(545, roi-10),
					font,
					0.6,
					(0, 0, 0xFF),
					2,
					cv2.LINE_AA,
					)

				output_video.write(frame)
				print ('writing frame ' + str(frame_counter) + '/' + str(total_frame))

				if frame_counter % (interval*fps) == 0:
					current_row = frame_counter//(interval*fps)
					worksheet.write(current_row, 0, frame_counter//fps)
					worksheet.write(current_row, 1, total_passed_object_per_interval)
					if current_row == 1:
						worksheet.write(current_row, 2, '=B2')
					else:
						worksheet.write(current_row, 2, '=B' + str(current_row+1) + '+C' + str(current_row)) # =B(x+1)+C(x)
					total_passed_object_per_interval = 0

				# # print for last frame
				# if frame_counter == total_frame:
				# 	worksheet.write(current_row, 0, frame_counter//fps)
				# 	worksheet.write(current_row, 1, total_passed_object_per_interval)
				# 	if current_row == 1:
				# 		worksheet.write(current_row, 2, '=B' + str(current_row+1))
				# 	else:
				# 		worksheet.write(current_row, 2, '=B' + str(current_row+1) + '+C' + str(current_row))
				
				final_col = current_row

				frame_counter += 1

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break

			# crate the graph consisting of bar and line chart
			bar_chart = workbook.add_chart({'type':'column'})
			bar_chart.add_series({
				'name':'=Sheet1!B1',
				'categories':'=Sheet1!A2:A' + str(final_col+1),
				'values':'=Sheet1!B2:B' + str(final_col+1)
				})
			line_chart = workbook.add_chart({'type':'line'})
			line_chart.add_series({
				'name':'=Sheet1!C1',
				'categories':'=Sheet1!A2:A' + str(final_col+1),
				'values':'=Sheet1!C2:C' + str(final_col+1)
				})
			bar_chart.combine(line_chart)

			bar_chart.set_title({'name':'No of Pedestrians'})
			bar_chart.set_x_axis({'name':'=Sheet1!A1'})
			bar_chart.set_y_axis({'name':'Pedestrians'})
			worksheet.insert_chart('F2', bar_chart)

			# mode, median and mean of data (count)
			worksheet.write('A' + str(final_col+3), 'MODE', bold)
			worksheet.write('A' + str(final_col+4), 'MEDIAN', bold)
			worksheet.write('A' + str(final_col+5), 'MEAN', bold)
			worksheet.write('A' + str(final_col+6), 'SD', bold)
			worksheet.write('B' + str(final_col+3), '=MODE(B2:B' + str(final_col+1) + ')')
			worksheet.write('B' + str(final_col+4), '=MEDIAN(B2:B' + str(final_col+1) + ')')
			worksheet.write('B' + str(final_col+5), '=AVERAGE(B2:B' + str(final_col+1) + ')')
			worksheet.write('B' + str(final_col+6), '=STDEV(B2:B' + str(final_col+1) + ')')
			worksheet.write('C' + str(final_col+3), '=MODE(C2:C' + str(final_col+1) + ')')
			worksheet.write('C' + str(final_col+4), '=MEDIAN(C2:C' + str(final_col+1) + ')')
			worksheet.write('C' + str(final_col+5), '=AVERAGE(C2:C' + str(final_col+1) + ')')
			worksheet.write('C' + str(final_col+6), '=STDEV(C2:C' + str(final_col+1) + ')')

			workbook.close()

			cap.release()
			cv2.destroyAllWindows()