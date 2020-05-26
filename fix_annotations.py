#!/usr/bin/env python3

# Always import these
import os
import sys
from ast import literal_eval
#from mylibs.collections import DotDict, ensure_iterable
#from mylibs.general import make_module_callable
import argparse
from tkinter import *
from tkinter.colorchooser import askcolor
import tkinter.filedialog as filedialog
from joblib.parallel import Parallel, delayed

# Import whatever else is needed
import cv2
import numpy as np


# Constants
IMG_EXTS = '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'
VALID_RE = {
	'SBVPI': re.compile(r'(?P<id>\d+)(?P<eye>[LR])_(?P<gaze>[lrsu])_(?P<n>\d+)'),
	'MOBIUS': re.compile(r'(?P<id>\d+)_(?P<phone>\d+)(?P<light>[nip])_(?P<eye>[LR])(?P<gaze>[lrsu])_(?P<n>(\d+)|(bad))')
}


class Args(argparse.Namespace):
	def __init__(self):
		# Default values
		#self.annotations = os.path.join(get_eyez_dir(), 'MOBIUS', 'Annotations Test')
		#self.original = os.path.join(get_eyez_dir(), 'MOBIUS', 'Images')
		#self.target = os.path.join(get_eyez_dir(), 'MOBIUS', 'Fixed Annotations')
		#self.channels = {'sclera': (0, 255, 0), 'iris': (255, 0, 0), 'pupil': (0, 0, 255)}
		self.annotations = r'C:\path\to\annotations'
		self.original = r'C:\path\to\original'
		self.target = r'C:\path\to\save\fixed\annotations\to'

		# Extra keyword arguments
		self.extra = DotDict()


def main(args):
	if not os.path.isdir(args.annotations):
		raise ValueError(f"{args.annotations} is not a directory.")
	if not os.path.isdir(args.original):
		raise ValueError(f"{args.original} is not a directory.")

	strict = args.extra.get('strict_match', False)
	exts = tuple(map(str.lower, ensure_iterable(args.extra.get('valid_ext', IMG_EXTS), True)))
	valid_re = [re.compile(s) for s in ensure_iterable(args.extra.valid_re, True)] if 'valid_re' in args.extra else VALID_RE.values()

	# Filter out unmatching names
	source_fs = {os.path.splitext(fname)[0]: os.path.join(root, fname) for root, _, fnames in os.walk(args.annotations) for fname in fnames if _match(fname, strict, exts, valid_re)}
	# This dictionary approach is the fastest way to handle this intersection without assuming the same directory structure
	original_fs = {os.path.splitext(fname)[0]: os.path.join(root, fname) for root, _, fnames in os.walk(args.original) for fname in fnames if os.path.splitext(fname)[0] in source_fs}
	# This still assumes all annotations have corresponding original images present
	source_fs, original_fs = zip(*((source_fs[basename], original_fs[basename]) for basename in source_fs))

	# Handle tree structure keeping
	if args.extra.get('keep_structure', True):
		target_fs = [os.path.join(args.target, os.path.relpath(f, args.annotations)) for f in source_fs]
	else:
		target_fs = [os.path.join(args.target, os.path.basename(f)) for f in source_fs]

	# If we're not overwriting, we need to filter out existing files
	# See https://stackoverflow.com/questions/17995302/filtering-two-lists-simultaneously
	if not args.extra.get('overwrite', True):
		source_fs, original_fs, target_fs = zip(*(fs for fs in zip(source_fs, original_fs, target_fs) if os.path.isfile(fs[2])))

	# Make necessary dirs so we don't run into 'path does not exist' errors
	for target_f in target_fs:
		os.makedirs(os.path.dirname(target_f), exist_ok=True)

	Parallel(n_jobs=-1)(
		delayed(_process_image)(source_f, original_f, target_f)
		for source_f, original_f, target_f in zip(source_fs, original_fs, target_fs)
	)


def _get_contour_center(contour):
	m = cv2.moments(contour)
	cx = int(m["m10"] / m["m00"])
	cy = int(m["m01"] / m["m00"])
	return cx, cy


def _get_contour_coordinates(contour):
	coordinates = list()
	approx_biggest = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True)
	n = approx_biggest.ravel()
	i = 0
	for j in n:
		if i % 2 == 0:
			x = n[i]
			y = n[i + 1]
			coordinates.append((x, y))
		i += 1
	return coordinates


def _process_image(annotation_f, original_f, target_f):
	basename = os.path.splitext(os.path.basename(annotation_f))[0]
	print(f"Processing {basename}")
	assert os.path.splitext(os.path.basename(original_f))[0] == basename

	try:
		# Parameters
		image = cv2.imread(annotation_f)
		output = target_f
		original = cv2.imread(original_f.replace('.png', '.jpg'))

		# Load image
		img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

		# Find red contours
		mask_red1 = cv2.inRange(img_hsv, np.array([0, 245, 245]), np.array([0, 255, 255]))
		mask_red2 = cv2.inRange(img_hsv, np.array([0, 255, 255]), np.array([0, 255, 255]))
		mask_red = mask_red1 + mask_red2
		contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		biggest_contour_red = max(contours_red, key=cv2.contourArea)

		# Find green contours
		mask_green = cv2.inRange(img_hsv, np.array([50, 50, 120]), np.array([70, 255, 255]))
		contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		biggest_contour_green = max(contours_green, key=cv2.contourArea)
		# Create list of contours to fill inside
		contours_green_final = list()
		if cv2.contourArea(biggest_contour_green) > 5000:
			contours_green_final.append(biggest_contour_green)
		# Get coordinates of the biggest green contour
		coordinates_green_biggest = _get_contour_coordinates(biggest_contour_green)
		coordinates_green_biggest = sorted(coordinates_green_biggest, key=lambda tup: tup[1])
		top_green_biggest = coordinates_green_biggest[0]
		bottom_green_biggest = coordinates_green_biggest[-1]
		# Draw all green contours but the biggest one on the mask
		for c in contours_green:
			if not np.array_equal(biggest_contour_green, c):
				# Get coordinates of other contours
				coordinates_green = _get_contour_coordinates(c)
				coordinates_green = sorted(coordinates_green, key=lambda tup: tup[1])
				top_green_other = coordinates_green[0]
				bottom_green_other = coordinates_green[-1]
				# Check that coordinates match
				if bottom_green_other[1] < top_green_biggest[1] or top_green_other[1] > bottom_green_biggest[1]:
					continue
				else:
					contours_green_final.append(c)

		# Find blue contours
		mask_blue = cv2.inRange(img_hsv, np.array([110, 255, 255]), np.array([130, 255, 255]))
		contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		biggest_contour_blue = max(contours_blue, key=cv2.contourArea)

		# Fill missing edges
		red_center = _get_contour_center(biggest_contour_red)
		hull_green = []
		for cnt in contours_green_final:
			hull_green.append(cv2.convexHull(cnt, False))
		cv2.drawContours(original, hull_green, -1, (0, 255, 0), -1)
		if cv2.contourArea(biggest_contour_red) > 2500 and (top_green_biggest[1] < red_center[1] < bottom_green_biggest[1]):
			hull_red = cv2.convexHull(biggest_contour_red, False)
			cv2.drawContours(original, [hull_red], -1, (0, 0, 255), -1)

		# Fill inside of final contours with correct color
		cv2.drawContours(original, contours_green_final, -1, (0, 255, 0), -1)
		if cv2.contourArea(biggest_contour_red) > 2500 and (top_green_biggest[1] < red_center[1] < bottom_green_biggest[1]):
			cv2.drawContours(original, [biggest_contour_red], -1, (0, 0, 255), -1)
		blue_center = _get_contour_center(biggest_contour_blue)
		if cv2.contourArea(biggest_contour_blue) > 2500 and (top_green_biggest[1] < blue_center[1] < bottom_green_biggest[1]):
			cv2.drawContours(original, [biggest_contour_blue], -1, (255, 0, 0), -1)

		# Write to original image
		print(f"Saving to {target_f}")
		cv2.imwrite(output, original)

	except (cv2.error, ValueError):
		print('Napaka v ' + basename)


def _match(f, strict, valid_exts, valid_re):
	basename, ext = os.path.splitext(os.path.basename(f))
	if not ext.lower() in valid_exts:
		return None
	for regex in valid_re:
		match = regex.fullmatch(basename) if strict else regex.match(basename)
		if match:
			return match
	return None

def process_command_line_options():
	args = Args()

	ap = argparse.ArgumentParser(description="Plot distribution histograms.")
	ap.add_argument('annotations', nargs='?', default=args.annotations, help="directory with annotations")
	ap.add_argument('original', nargs='?', default=args.original, help="directory with original images")
	ap.add_argument('target', nargs='?', default=args.target, help="directory to save fixed annotations to")
	#ap.add_argument('-c', '--channel', action='append', nargs=2, default=[], help="channel-colour mapping pair (specified as 'channel r,g,b')")
	ap.parse_known_args(namespace=args)

	ap = argparse.ArgumentParser(description="Extra keyword arguments.")
	ap.add_argument('--flatten', action='store_false', dest='keep_structure', help="flatten directory tree structure")
	ap.add_argument('--no-overwrite', action='store_false', dest='overwrite', help="do not overwrite files that already exist in destination")
	ap.add_argument('--strict-match', action='store_true', help="use strict matching in file name filtering")
	ap.add_argument('-e', '--extra', nargs=2, action='append', default=[], help="any extra keyword-value argument pairs")
	ap.parse_known_args(namespace=args.extra)

	for key, value in args.extra.extra:
		try:
			args.extra[key] = literal_eval(value)
		except ValueError:
			args.extra[key] = value
	del args.extra['extra']

	'''
	if args.channel:
		args.channels = {channel.lower(): literal_eval(colour) for channel, colour in args.channel}
		del args.channel
	'''

	return args


class GUI(Tk):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)
		self.title("Fix Annotations")
		self.args = Args()
		self.ok = False

		self.frame = Frame(self)
		self.frame.pack(fill=BOTH, expand=YES)

		# In grid(), column default is 0, but row default is first empty row.
		row = 0
		self.source_lbl = Label(self.frame, text="Annotations:")
		self.source_lbl.grid(column=0, row=row, sticky='w')
		self.source_txt = Entry(self.frame, width=60)
		self.source_txt.insert(END, self.args.annotations)
		self.source_txt.grid(column=1, row=row)
		self.source_btn = Button(self.frame, text="Browse", command=self.browse_source)
		self.source_btn.grid(column=2, row=row)

		row += 1
		self.original_lbl = Label(self.frame, text="Original:")
		self.original_lbl.grid(column=0, row=row, sticky='w')
		self.original_txt = Entry(self.frame, width=60)
		self.original_txt.insert(END, self.args.original)
		self.original_txt.grid(column=1, row=row)
		self.original_btn = Button(self.frame, text="Browse", command=self.browse_original)
		self.original_btn.grid(column=2, row=row)

		row += 1
		self.target_lbl = Label(self.frame, text="Target:")
		self.target_lbl.grid(column=0, row=row, sticky='w')
		self.target_txt = Entry(self.frame, width=60)
		self.target_txt.insert(END, self.args.target)
		self.target_txt.grid(column=1, row=row)
		self.target_btn = Button(self.frame, text="Browse", command=self.browse_target)
		self.target_btn.grid(column=2, row=row)

		'''
		row += 1
		self.channel_frame = ExtraFrame(self.frame, frame_type=CCFrame)
		self.channel_frame.grid(column=0, row=row, columnspan=3, sticky='w')
		self.channel_frame.key_lbl['text'] = "Channel"
		self.channel_frame.value_lbl['text'] = "Colour"
		for channel, colour in self.args.channels.items():
			self.channel_frame.add_pair(channel, colour)
		'''

		row += 1
		self.extra_frame = ExtraFrame(self.frame)
		self.extra_frame.grid(row=row, columnspan=3, sticky='w')

		row += 1
		self.ok_btn = Button(self.frame, text="OK", command=self.confirm)
		self.ok_btn.grid(column=1, row=row)
		self.ok_btn.focus()

	def browse_source(self):
		self._browse_dir(self.source_txt)

	def browse_original(self):
		self._browse_dir(self.original_txt)

	def browse_target(self):
		self._browse_dir(self.target_txt)

	def _browse_dir(self, target_txt):
		init_dir = target_txt.get()
		while not os.path.isdir(init_dir):
			init_dir = os.path.dirname(init_dir)

		new_entry = filedialog.askdirectory(parent=self, initialdir=init_dir)
		if new_entry:
			_set_entry_text(target_txt, new_entry)

	def _browse_file(self, target_txt, exts=None):
		init_dir = os.path.dirname(target_txt.get())
		while not os.path.isdir(init_dir):
			init_dir = os.path.dirname(init_dir)

		if exts:
			new_entry = filedialog.askopenfilename(parent=self, filetypes=exts, initialdir=init_dir)
		else:
			new_entry = filedialog.askopenfilename(parent=self, initialdir=init_dir)

		if new_entry:
			_set_entry_text(target_txt, new_entry)

	def confirm(self):
		self.args.annotations = self.source_txt.get()
		self.args.original = self.original_txt.get()
		self.args.target = self.target_txt.get()
		#self.args.channels = self._parse_channel_entries()

		for kw in self.extra_frame.pairs:
			key, value = kw.key_txt.get(), kw.value_txt.get()
			if key:
				try:
					self.args.extra[key] = literal_eval(value)
				except ValueError:
					self.args.extra[key] = value

		self.ok = True
		self.destroy()
	
	def _parse_channel_entries(self):
		return {pair.key_txt.get().lower(): _parse_colour(pair.value_txt.get()) for pair in self.channel_frame.pairs if pair.key_txt.get()}


class ExtraFrame(Frame):
	def __init__(self, *args, frame_type=None, **kw):
		super().__init__(*args, **kw)
		self._frame_type = frame_type if frame_type else KVFrame
		self.pairs = []

		self.key_lbl = Label(self, width=30, text="Key", anchor='w')
		self.value_lbl = Label(self, width=30, text="Value", anchor='w')

		self.add_btn = Button(self, text="+", command=self.add_pair)
		self.add_btn.grid()

	def add_pair(self, key=None, val=None):
		pair_frame = self._frame_type(self, pady=2)
		if key:
			pair_frame.key_txt.insert(END, key)
		if val:
			pair_frame.value_txt.insert(END, val)

		self.pairs.append(pair_frame)
		pair_frame.grid(row=len(self.pairs), columnspan=3)
		self.update_labels_and_button()

	def update(self, indices=None):
		for i, pair in enumerate(self.pairs):
			if indices and i not in indices:
				continue
			pair.grid(row=i+1)
		self.update_labels_and_button()

	def update_labels_and_button(self):
		if self.pairs:
			self.key_lbl.grid(column=0, row=0, sticky='w')
			self.value_lbl.grid(column=1, row=0, sticky='w')
		else:
			self.key_lbl.grid_remove()
			self.value_lbl.grid_remove()
		self.add_btn.grid(row=len(self.pairs) + 1)


class KVFrame(Frame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

		self.key_txt = Entry(self, width=30)
		self.key_txt.grid(column=0, row=0)

		self.value_txt = Entry(self, width=30)
		self.value_txt.grid(column=1, row=0)

		self.up_btn = Button(self, text="^", command=self.move_up)
		self.up_btn.grid(column=2, row=0)

		self.down_btn = Button(self, text="v", command=self.move_down)
		self.down_btn.grid(column=3, row=0)

		self.remove_btn = Button(self, text="-", command=self.remove)
		self.remove_btn.grid(column=4, row=0)

	def remove(self):
		i = self.master.pairs.index(self)
		del self.master.pairs[i]
		self.master.update(range(i, len(self.master.pairs)))
		self.destroy()

	def move_up(self):
		i = self.master.pairs.index(self)
		if i == 0:
			return
		self.master.pairs[i-1], self.master.pairs[i] = self.master.pairs[i], self.master.pairs[i-1]
		self.master.update((i-1, i))

	def move_down(self):
		i = self.master.pairs.index(self)
		if i == len(self.master.pairs) - 1:
			return
		self.master.pairs[i], self.master.pairs[i+1] = self.master.pairs[i+1], self.master.pairs[i]
		self.master.update((i, i+1))


class CCFrame(KVFrame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

		self.colour_btn = Button(self, text="Colour", command=self.pick_colour)
		self.colour_btn.grid(column=2, row=0)

		self.up_btn.grid(column=3, row=0)
		self.down_btn.grid(column=4, row=0)
		self.remove_btn.grid(column=5, row=0)

	def pick_colour(self):
		try:
			colour = askcolor(parent=self.master.master, initialcolor=_parse_colour(self.value_txt.get()))[0]
		except TypeError:
			colour = askcolor(parent=self.master.master, initialcolor=(0, 255, 0))[0]
		if colour:
			_set_entry_text(self.value_txt, " ".join(map(str, map(int, colour))))


def _set_entry_text(entry, txt):
	entry.delete(0, END)
	entry.insert(END, txt)


def _parse_colour(txt):
	return tuple(map(int, re.findall(r'\d+', txt)))


def ensure_iterable(x, listify_single_string=False):
	if isinstance(x, str) and listify_single_string:
		return [x]
	try:
		_ = iter(x)
		return x
	except TypeError:
		try:
			for _ in x:
				pass
			return x
		except TypeError:
			return [x]


class DotDict(dict):
	def __init__(self, *args, **kwargs):
		d = dict(*args, **kwargs)
		for key, val in d.items():
			if isinstance(val, Mapping):
			    value = DotDict(val)
			else:
			    value = val
			self[key] = value

	def __delattr__(self, name):
		try:
			del self[name]
		except KeyError as ex:
			raise AttributeError(f"No attribute called: {name}") from ex

	def __getattr__(self, k):
		try:
			return self[k]
		except KeyError as ex:
			raise AttributeError(f"No attribute called: {k}") from ex

	__setattr__ = dict.__setitem__


if __name__ == '__main__':
	# If CLI arguments, read them
	if len(sys.argv) > 1:
		args = process_command_line_options()

	# Otherwise get them from a GUI
	else:
		gui = GUI()
		gui.mainloop()
		# If GUI was exited but not with the OK button
		if not gui.ok:
			sys.exit(0)
		args = gui.args

	main(args)

#else:
#	# Make module callable (python>=3.5)
#	make_module_callable(__name__, main)
