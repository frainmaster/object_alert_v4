#!/usr/bin/python3.7

#----------------------------------------------
#--- Author         : Irfan Ahmad
#--- Mail           : irfanibnuahmad@gmail.com
#--- Date           : 11th July 2020
#----------------------------------------------

import os
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from tkinter import scrolledtext
import cv2
import json

class ObjectAlertApp(Tk):
	def __init__(self):
		Tk.__init__(self)
		self.title('Object Alert')
		# self.resizable(False, False)
		self.geometry('1080x640')
		self.cap_width = 0
		self.cap_height = 0

		# ATTRIBUTES
		# model dropdown
		# self.models = [
		# 	{'name':'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03', 'speed':'26'},
		# 	{'name':'ssd_mobilenet_v1_coco_2018_01_28', 'speed':'30'},
		# 	{'name':'faster_rcnn_inception_v2_coco_2018_01_28', 'speed':'58'}
		# ]
		self.models = os.listdir('detection_model')

		# canvas
		self.start_x = self.start_y = 0
		self.end_x = self.end_y = 0
		self.previous_x = self.previous_y = 0
		self.x = self.y = 0

		# COMPONENTS
		# title label
		self.lbl = Label(text = 'Object Alert', font = ('Arial', 12))
		self.lbl.grid(row = 0, column = 1, columnspan = 5)
		self.grid_rowconfigure(0, minsize = 40)

		# detection model dropdown
		self.model_label = Label(text = 'Detection model')
		self.model_label.grid(row = 2, column = 0)
		self.grid_columnconfigure(0, minsize = 150)

		self.model_name = StringVar()
		# self.model_name.set(self.getModelString(self.models[0]))
		self.model_name.set(self.models[1])

		self.model_dropdown = OptionMenu(
			self,
			self.model_name,
			# *[self.getModelString(model) for model in self.models],
			*[model for model in self.models],
			)
		self.model_dropdown.config(width=40, font=('Helvetica', 8))
		self.model_dropdown.grid(row = 2, column = 1, columnspan = 4)
		self.grid_rowconfigure(2, minsize = 40)

		self.video_label = Label(text = 'Video')
		self.video_label.grid(row = 3, column = 0)

		# select video button
		self.btn_video = Button(
			self,
			text = 'Select video',
			bg = 'white',
			fg = 'black',
			command = self.selectVideo,
			font = ('Arial',10))
		self.btn_video.grid(row = 3, column = 1)

		self.video_name = Label(text = 'No videos selected')
		self.video_name.grid(row = 3, column = 2, columnspan = 2)
		self.grid_rowconfigure(3, minsize = 40)
		self.selected_video = Label(text = '')

		# canvas
		self.canvas_bg_img = PhotoImage(file = 'input/res/buhlack.png')
		self.white_bg = PhotoImage(file = 'input/res/wite.png')
		self.img_width = self.canvas_bg_img.width()
		self.img_height = self.canvas_bg_img.height()
		self.canvas = Canvas(
			self,
			width = self.img_width,
			height = self.img_height,
			cursor = 'cross')
		self.canvas.create_image(0, 0, image = self.canvas_bg_img, anchor = 'nw')
		# self.canvas.grid(row = 4, column = 0, columnspan = 10)
		self.canvas.grid(row = 2, column = 6, rowspan = 7)
		self.btn_rotate = Button(
			self,
			width = 15,
			text = 'Rotate',
			command = self.rotate_vid)
		self.btn_get_points = Button(
			self,
			width = 15,
			text = 'Get points',
			command = self.print_points)
		# self.btn_get_points.grid(row = 5, column = 2)
		self.btn_clear_canvas = Button(
			self,
			width = 15,
			text = 'Clear',
			command = self.clear_canvas)
		# self.btn_clear_canvas.grid(row = 5, column = 3)
		self.grid_rowconfigure(5, minsize = 50)
		# self.canvas.bind('<Button-1>', self.start_point)
		# self.canvas.bind('<B1-Motion>', self.end_point)

		# log
		self.activity_log = scrolledtext.ScrolledText(self, width = 50, height = 10)
		self.activity_log.grid(row = 6, column = 0, columnspan = 5)
		self.activity_log.insert(INSERT, 'Activity log\n')
		self.activity_log.configure(state = DISABLED)

		# submit button
		self.btn_submit = Button(
			self,
			text = 'Display data',
			bg = 'white',
			fg = 'black',
			height = 3,
			command = self.submitData)
		self.btn_submit.grid(row = 6, column = 5)

	getFileNameWithoutPath = lambda self, file : file[file.rfind('/')+1:]
	# getModelString = lambda self, model : model['name'] + ' - ' + model['speed'] + 'ms'
	# getModelString = lambda self, model : model['name']

	def getProperFileName(self, filename):
		file_ext = filename[filename.rfind('.'):]
		filename = filename[filename.rfind('/')+1:].replace(file_ext, '') # remove file path and extension
		return (filename[:28] + '...' + file_ext) if len(filename) > 30 else filename + file_ext

	def selectVideo(self):
		filename = filedialog.askopenfilename(filetypes = (('Video', '*.mp4 *.avi'), ('AVI', '*.avi')))
		self.selected_video.config(text = self.getFileNameWithoutPath(filename))
		if self.getProperFileName(filename) == '':
			self.video_name.config(text = 'No videos selected')
		else:
			self.video_name.config(text = self.getProperFileName(filename))
			# enable drawing box only after video is selected
			self.canvas.bind('<Button-1>', self.start_point)
			self.canvas.bind('<B1-Motion>', self.end_point)
			self.btn_rotate.grid(row = 5, column = 1)
			self.btn_get_points.grid(row = 5, column = 2)
			self.btn_clear_canvas.grid(row = 5, column = 3)
			cap = cv2.VideoCapture('input/' + self.getFileNameWithoutPath(filename))
			self.cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			self.cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			ret, frame = cap.read()
			cv2.imwrite('input/res/img.png', frame)
			self.canvas.config(width = self.cap_width, height = self.cap_height)
			self.new_bg_img = PhotoImage(file = 'input/res/img.png')
			self.canvas.create_image(0, 0, image = self.new_bg_img, anchor = 'nw')
			self.log_msg('Selected video ' + self.getFileNameWithoutPath(filename))

	def submitData(self):
		input_video = self.selected_video.cget('text')
		input_video = '' if input_video == 'No videos selected' else input_video
		selected_model = self.model_name.get()
		# selected_model = selected_model[:selected_model.rfind('-')-1]
		if selected_model == '' or selected_model == None:
			messagebox.showinfo('Error', 'Please select a detection model')
		elif input_video == '' or input_video == None:
			messagebox.showinfo('Error', 'Please select an input video')
		else:
			if messagebox.askyesno(
				'Generate video?',
				'Detection model: ' + selected_model
				+ '\nSelected video: ' + input_video):
				start_point, end_point = (self.start_x, self.start_y), (self.x, self.y)
				roi = (start_point[0]+end_point[0])//2
				if messagebox.askyesno(
					'Save points?',
					'Do you want to save the start and end points as default points afterwards?'):
					self.saveToJson(input_video, str(start_point), str(end_point))
				from utils import backbone
				from api import object_alert_api as oa_api
				self.log_msg('selected_model: ' + selected_model)
				self.log_msg('input_video: ' + input_video)
				detection_graph, category_index = backbone.set_model(selected_model, 'mscoco_label_map.pbtxt')
				self.log_msg('Video is being generated')
				oa_api.object_alert(
					'input/' + input_video,
					detection_graph,
					category_index,
					start_point,
					end_point,
					roi)
				self.log_msg('Video and excel file has been produced')
			else:
				self.log_msg('Generate video cancelled')

	def saveToJson(self, input_video, start_point, end_point):
		video_data = './input/res/video_data.json'
		f = open(video_data)
		data = json.load(f)
		f.close()
		new_data = {'start_point': start_point[1:len(start_point)-1], 'end_point': end_point[1:len(end_point)-1]}
		data['data'][input_video] = new_data
		f = open(video_data, 'w')
		f.write(json.dumps(data, indent=4))
		f.close()
		print('data for {} is updated'.format(input_video))

	# canvas functions
	def clear_canvas(self):
		self.start_x = self.start_y = self.x = self.y = 0
		self.canvas.delete('box')

	erase_box = lambda self : self.canvas.delete('box')

	def print_points(self):
		self.log_msg('point: ' + str([self.start_x, self.start_y, self.x, self.y]))

	def rotate_vid(self):
		img = cv2.imread('input/res/img.png')
		img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
		cv2.imwrite('input/res/img.png', img)
		self.cap_width, self.cap_height = self.cap_height, self.cap_width
		self.canvas.config(width = self.cap_width, height = self.cap_height)
		self.new_bg_img = PhotoImage(file = 'input/res/img.png')
		self.canvas.create_image(0, 0, image = self.new_bg_img, anchor = 'nw')

	def end_point(self, event):
		self.erase_box()
		self.x = event.x
		self.y = event.y
		self.canvas.create_rectangle(
			self.start_x,
			self.start_y,
			self.x,
			self.y,
			outline = 'black',
			tag='box')
		self.previous_x = self.x
		self.previous_y = self.y

	def start_point(self, event):
		self.erase_box()
		self.start_x = event.x
		self.start_y = event.y

	# log function
	def log_msg(self, message):
		self.activity_log.configure(state = NORMAL)
		self.activity_log.insert(INSERT, message + '\n')
		self.activity_log.configure(state = DISABLED)

if __name__ == '__main__':
	app = ObjectAlertApp()
	app.mainloop()