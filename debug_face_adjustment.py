# test_face_adjustment.py
# This is test code to alter an attacker's image to spoof a victim.
# Make sure that images provided as input only have one face in them.
# The code will still work but you won't have control over which face
# is chosen to compare/alter

# example usage:
# python test_face_adjustment.py --encodings encodings.pickle --image1 michael_pics/alec1.jpg --image2 michael_pics/adrian1.jpg

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import cv2
import math
import numpy
import statistics
from shapely.geometry import LineString
import copy

# size of the picture frame (image_size x image_size)
image_size = 512
# how much to extend the image bounds returned by face capture.  The
# face_recognition.face_locations() function will find a face but
# sometimes features are cut off.  A value is chosen by trial and
# error to include as many features in the picture as possible.
boundary_extension = 50
# path to output images.  Used for debugging
out_path = "test_pic_output/"
# size to extend to grab all eye features
eye_bounds = 10

def show_image(name, image):
	print("Press any key to continue")
	cv2.imshow(name, image)
	cv2.waitKey()
	#cv2.destroyAllWindows()

def save_image(name, image):
	print("[INFO] image saved to " + out_path + name + ".jpg")
	cv2.imwrite(out_path + name + ".jpg", image)

# This function marks the locations of the face for face identification.
# The red dots are part of the 68 point ID and the blue point is the
# center of the image
def mark_faces(image):
	image = copy.deepcopy(image)
	landmarks = face_recognition.face_landmarks(image)[0]
	#print(landmarks)

	for key in landmarks:
		for i in landmarks[key]:
			cv2.circle(image, i, 2, (0,0,255), 2)

	cv2.circle(image, (int(image_size/2), int(image_size/2)), 2, (255,0,0), 2)
	
	#show_image("marked_image", image)

	return image

# Returns the color at a certain point on an image
def get_color(image, point):
	x, y = point
	return image[y][x]

# This function translates the image so that the nose is in the center of
# the frame.  For example, if the boundary_extension variable is set to 300,
# the nose will be set to position (150, 150)  This is needed to rotate the
# image properly using the eyes as reference points
def translate_face(image):
	#cv2.imwrite(out_path + "pre-translate.jpg", image)
	#show_image("pre-translate", image)

	rows,cols = image.shape[:-1]

	landmarks = face_recognition.face_landmarks(image)[0]
	nose_center = landmarks['nose_bridge'][-1]
	x = image_size/2 - nose_center[0]
	y = image_size/2 - nose_center[1]

	M = numpy.float32([[1,0,x],[0,1,y]])

	translated_image =  cv2.warpAffine(image, M, (cols,rows))

	#cv2.imwrite(out_path + "post-translate.jpg", image)
	#show_image("post-translate", translated_image)

	return translated_image

# This function will rotate the face so that it is level with the camera.
# The y delta is used to determine how off-balanced the image is and the
# right eye is used with trigonometric functions to determine how much
# rotation is needed.  The output is not perfect so it is recommended to
# run at lease 3 iterations until the face is level
def rotate_face(image, iterations=3):

	while iterations:

		landmarks = face_recognition.face_landmarks(image)[0]
		right_eye = landmarks['right_eye']
		left_eye = landmarks['left_eye']

		rx = 0
		ry = 0
		for i in right_eye:
			rx += i[0]
			ry += i[1]
		rx = rx/len(right_eye)
		ry = ry/len(right_eye)
		
		lx = 0
		ly = 0
		for i in left_eye:
			lx += i[0]
			ly += i[1]
		lx = lx/len(left_eye)
		ly = ly/len(left_eye)
		
		vert_average = (ry + ly)/2
		hypotenuse = math.sqrt((rx - image_size/2)**2 + ((ry - image_size/2) * -1)**2)
		orig_deg = math.degrees(math.atan(((ry - image_size/2) * -1)/(rx - image_size/2)))
		needed_deg = math.degrees(math.asin(((vert_average - image_size/2) * -1)/hypotenuse))

		rotated_image = imutils.rotate(image, needed_deg - orig_deg)

		#cv2.imwrite(out_path + "pre-rotate.jpg", image)
		#show_image("pre-rotate", image)

		#cv2.imwrite(out_path + "post-rotate.jpg", rotated_image)
		#show_image("post-rotate", rotated_image)

		iterations -= 1
		image = rotated_image

	return image

# Find the euclidean distance between 2 points. point_a(x, y) and point_b(x, y)
def find_distance(point_a, point_b):
	dist = 0

	dist = math.sqrt((point_a[0] - point_b[0])**2 + (point_a[0] - point_b[0])**2)
	print("DISTANCE: " + str(dist))

	return dist

def process_face(image_file):
	print("[INFO] loading and processing " + image_file + "...")
	image = cv2.imread(image_file)
	image = imutils.resize(image, width=600)
	face_boundaries = face_recognition.face_locations(image)[0]
	image = cv2.resize(image[face_boundaries[0]-boundary_extension:face_boundaries[2]+boundary_extension, face_boundaries[3]-boundary_extension:face_boundaries[1]+boundary_extension], (image_size, image_size))
	image = translate_face(image)
	image = rotate_face(image)
	return image


# create a black boundary around the chin of the attacker image
def chin_blackout(attacker, victim_chin):
	v_chin_points = [[x,y] for x, y in victim_chin]
	v_chin_points += [[image_size, 0], [image_size, image_size], [0, image_size], [0, 0]]
	poly = numpy.array( [v_chin_points], dtype=numpy.int32 )
	cv2.fillPoly(attacker, poly, (255,255,255))


def chin_overflow(box, color_ref_point):
	delta_x = box['a'][0] - box['v'][0]
	delta_y = box['a'][1] - box['v'][1]

	if delta_x > 0:
		box['a'][0] += 10
		color_ref_point[0] += 10
	elif delta_x < 0:
		box['a'][0] -= 10
		color_ref_point[0] -= 10
	if delta_y > 0:
		box['a'][1] += 10
		color_ref_point[1] += 10
	elif delta_y < 0:
		box['a'][1] -= 10
		color_ref_point[1] -= 10
		
	return (box, color_ref_point)

# modify the attacker's chin to match the victim's shape
# For each iteration, a polygon in made from the following points [attacker-1, attacker, victim, victim-1]
# If the victim's chin is smaller, a black border is extended on the attacker's jaw to the edge of the victim's jawline position
# If the victim's chin is larger, a color is found halfway between the points attacker and attacker-1 on the attacker's jaw.  This color is extended to the victim's chin shape
def chin_adjust(attacker, victim, attacker_chin, victim_chin):
	print("[INFO] Adjusting Chin...")

	# last_attacker_ref_point is the last adjusted position for the attacker's chin
	last_attacker_ref_point = []
	# last_attacker_ref_point_orig is the last original position for the attacker's chin
	last_attacker_ref_point_orig = []
	# last_victim_ref_point is the last adjusted position for the victim's chin
	last_victim_ref_point = []

	# loop through every point of the chin to make adjustments
	for point in range(len(attacker_chin)):

		# box contains a dictionary of the attacker's chin points and victim's chin points
		box = {'a': list(attacker_chin[point]), 'v': list(victim_chin[point])}

		# for the first point, we do not have a last reference.  This iteration just finds the points on the attacker's face and adjusts
		if point == 0:
			#last_attacker_ref_point_orig = [box['a'][0], box['a'][1]]
			last_attacker_ref_point_orig = box['a']
			box, _ = chin_overflow(box, [0,0])
			last_attacker_ref_point = box['a']
			last_victim_ref_point = victim_chin[point]
			continue

		# This point is found as the midpoint of the line between points attacker and attacker-1
		color_ref_point = [int(statistics.mean([attacker_chin[point][0], last_attacker_ref_point_orig[0]])), int(statistics.mean([attacker_chin[point][1], last_attacker_ref_point_orig[1]]))]

		# save the original attacker's last reference point, we need the non-adjusted point for the next calculation
		last_attacker_ref_point_orig = [box['a'][0], box['a'][1]]
		# chin_overflow returns the adjusted color_point_reference to get a better skin tone
		# If we pull the color from a point on the jawline, the color might not match the attacker's skin tone,
		# or it may not be entirely black, we need to call chin_overflow to move the reference point in or out to get a good color.
		box, color_ref_point = chin_overflow(box, color_ref_point)
		# TODO: need to fix this
		try:
			color_ref = attacker[color_ref_point[1]][color_ref_point[0]]
		except:
			color_ref = (0,0,0)

		# generate a polygon of the 4 points
		poly = numpy.array( [[last_attacker_ref_point, last_victim_ref_point, box['v'], box['a']]], dtype=numpy.int32 )
		color_ref = tuple(int(num) for num in color_ref)
		cv2.fillPoly(attacker, poly, tuple(color_ref))

		last_attacker_ref_point = box['a']
		last_victim_ref_point = victim_chin[point]


def eye_cut(attacker, attacker_eye):

	# turn tuple list into list list
	attacker_eye = [[x,y] for x,y in attacker_eye]

	# grab a neutral color to cover up old eyes
	color_ref = attacker[int((attacker_eye[4][1] + attacker_eye[5][1])/2) + eye_bounds*2][int((attacker_eye[4][0] + attacker_eye[5][0])/2)]

	are_pts = numpy.array(attacker_eye)

	# make a rectangle around all 6 eye points
	are_rect = cv2.boundingRect(are_pts)
	arx,ary,arw,arh = are_rect
	arx -= eye_bounds
	ary -= eye_bounds
	arw += (eye_bounds*2)
	arh += (eye_bounds*2)

	# crop the image to just the size of the rectangle
	attacker_croped_eye = attacker[ary:ary+arh, arx:arx+arw].copy()

	# get a rectangle of the attacker's old eye and cover up with a neutral color
	eye_poly = numpy.array( [[[arx, ary], [arx, ary + arh], [arx + arw, ary + arh], [arx + arw, ary]]], dtype=numpy.int32 )
	color_ref = tuple(int(num) for num in color_ref)
	cv2.fillPoly(attacker, eye_poly, color_ref)

	return attacker_croped_eye


def eye_paste(attacker, victim_eye, attacker_croped_eye):
	print("[INFO] Adjusting Eye...")

	# turn tuple list into list list
	victim_eye = [[x,y] for x,y in victim_eye]
	vre_pts = numpy.array(victim_eye)

	# find location of victim eye
	vre_rect = cv2.boundingRect(vre_pts)
	vrx,vry,vrw,vrh = vre_rect
	vrx -= eye_bounds
	vry -= eye_bounds
	vrw += (eye_bounds*2)
	vrh += (eye_bounds*2)

	# crop attacker eye to victim eye size
	attacker_croped_eye = cv2.resize(attacker_croped_eye, (vrw, vrh))

	# copy new eye onto attackers face
	attacker[vry:vry+vrh, vrx:vrx+vrw] = attacker_croped_eye


def nose_cut(attacker, attacker_nose_bridge, attacker_nose_tip):
	# turn tuple list into list list
	attacker_nose_bridge = [[x,y] for x,y in attacker_nose_bridge]
	attacker_nose_tip = [[x,y] for x,y in attacker_nose_tip]

	# get nose tip width
	a_nose_tip_width = attacker_nose_tip[4][0] - attacker_nose_tip[0][0]

	# we want to get a triangle cut of the nose bridge to copy into a new location
	# get nose bridge dimensions and modify dimensions to extend a bit past the nose tip in the x direction
	a_top_bridge = attacker_nose_bridge[0]
	a_left_bridge = [attacker_nose_tip[0][0] - int(a_nose_tip_width/4), attacker_nose_bridge[3][1]]
	a_right_bridge = [attacker_nose_tip[4][0] + int(a_nose_tip_width/4), attacker_nose_bridge[3][1]]

	anb_pts = numpy.array([a_top_bridge, a_left_bridge, a_right_bridge])

	# make a rectangle around the 3 nose bridge points
	anb_rect = cv2.boundingRect(anb_pts)
	anbx,anby,anbw,anbh = anb_rect

	# crop the image to just the size of the rectangle
	attacker_croped_nose_bridge = attacker[anby:anby+anbh, anbx:anbx+anbw].copy()

	# make mask of nose triangle and trim out unnecessary face features
	mask_pts = anb_pts - anb_pts.min(axis = 0)
	mask = numpy.zeros(attacker_croped_nose_bridge.shape[:2], numpy.uint8)
	cv2.drawContours(mask, [mask_pts], -1, (255,255,255), -1, cv2.LINE_AA)
	attacker_croped_nose_bridge_mask = cv2.bitwise_and(attacker_croped_nose_bridge, attacker_croped_nose_bridge, mask=mask)
	# still deciding if we need a white or black mask
	bg = numpy.ones_like(attacker_croped_nose_bridge, numpy.uint8)*255
	cv2.bitwise_not(bg, bg, mask=mask)
	attacker_croped_nose_bridge_mask += bg

	# do the same for the nose tip (this time a rectangle)
	# get nose tip dimensions and modify dimensions to extend a bit past the nose tip in the x direction
	a_top_left_tip = [attacker_nose_tip[0][0] - int(a_nose_tip_width/4), attacker_nose_bridge[3][1]]
	a_bottom_right_tip = [attacker_nose_tip[4][0] + int(a_nose_tip_width/4), attacker_nose_tip[3][1]]

	ant_pts = numpy.array([a_top_left_tip, a_bottom_right_tip])

	# make a rectangle around the 3 nose bridge points
	ant_rect = cv2.boundingRect(ant_pts)
	antx,anty,antw,anth = ant_rect

	# crop the image to just the size of the rectangle
	attacker_croped_nose_tip = attacker[anty:anty+anth, antx:antx+antw].copy()

	# get a polygon of the attacker's old nose and cover up with a neutral color
	nose_poly = numpy.array( [[a_top_bridge, a_left_bridge, [a_top_left_tip[0], a_bottom_right_tip[1]], a_bottom_right_tip, a_right_bridge]], dtype=numpy.int32 )
	color_ref = attacker[attacker_nose_bridge[0][1]][attacker_nose_bridge[0][0]]
	color_ref = tuple(int(num) for num in color_ref)
	cv2.fillPoly(attacker, nose_poly, color_ref)

	return (attacker_croped_nose_bridge, attacker_croped_nose_tip)


def nose_paste(attacker, victim_nose_bridge, victim_nose_tip, attacker_nose_bridge_cut, attacker_nose_tip_cut):
	print("[INFO] Adjusting Nose...")
	# turn tuple list into list list
	victim_nose_bridge = [[x,y] for x,y in victim_nose_bridge]
	victim_nose_tip = [[x,y] for x,y in victim_nose_tip]

	# Cover up attacker's old nose

	# get nose tip width
	v_nose_tip_width = victim_nose_tip[4][0] - victim_nose_tip[0][0]

	# we want to get a triangle cut of the nose bridge to copy into a new location
	# get nose bridge dimensions and modify dimensions to extend a bit past the nose tip in the x direction
	v_top_bridge = victim_nose_bridge[0]
	v_left_bridge = [victim_nose_tip[0][0] - int(v_nose_tip_width/4), victim_nose_bridge[3][1]]
	v_right_bridge = [victim_nose_tip[4][0] + int(v_nose_tip_width/4), victim_nose_bridge[3][1]]

	vnb_pts = numpy.array([v_top_bridge, v_left_bridge, v_right_bridge])

	# find location of victim nose bridge
	vnb_rect = cv2.boundingRect(vnb_pts)
	vnbx,vnby,vnbw,vnbh = vnb_rect

	# resize attacker nose bridge to victim nose bridge size
	attacker_nose_bridge_cut = cv2.resize(attacker_nose_bridge_cut, (vnbw, vnbh))
	#show_image("attacker_nose_bridge_cut", attacker_nose_bridge_cut)

	# create a mask of the new nose
	# Create region of interest on attacker image
	roi = attacker[vnby:vnby+vnbh, vnbx:vnbx+vnbw]

	# Trim nose triangle
	mask_pts = numpy.array([v_top_bridge, v_left_bridge, v_right_bridge])
	mask_pts = mask_pts - mask_pts.min(axis = 0)
	mask = numpy.zeros(attacker_nose_bridge_cut.shape[:2], numpy.uint8)
	cv2.drawContours(mask, [mask_pts], -1, (255,255,255), -1, cv2.LINE_AA)
	attacker_nose_bridge_cut = cv2.bitwise_and(attacker_nose_bridge_cut, attacker_nose_bridge_cut, mask=mask)
	
	# Create mask of nose and create inverse mask
	attacker_nose_bridge_cut_gray = cv2.cvtColor(attacker_nose_bridge_cut, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(attacker_nose_bridge_cut_gray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	
	# black-out area of nose in ROI
	attacker_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
	
	# take only region of nose from nose image
	attacker_nose_bridge_cut_fg = cv2.bitwise_and(attacker_nose_bridge_cut, attacker_nose_bridge_cut, mask=mask)
	
	# put nose in roi and modify main image
	attacker_nose_bridge_cut = cv2.add(attacker_bg, attacker_nose_bridge_cut_fg)
	
	# do the same for the nose tip (this time a rectangle)
	# get nose tip dimensions and modify dimensions to extend a bit past the nose tip in the x direction
	v_top_left_tip = [victim_nose_tip[0][0] - int(v_nose_tip_width/4), victim_nose_bridge[3][1]]
	v_bottom_right_tip = [victim_nose_tip[4][0] + int(v_nose_tip_width/4), victim_nose_tip[3][1]]

	vnt_pts = numpy.array([v_top_left_tip, v_bottom_right_tip])

	# find location of victim nose tip
	vnt_rect = cv2.boundingRect(vnt_pts)
	vntx,vnty,vntw,vnth = vnt_rect

	# resize attacker nose tip to victim nose tip size
	attacker_nose_tip_cut = cv2.resize(attacker_nose_tip_cut, (vntw, vnth))

	# Copy new nose onto attacker's face
	attacker[vnby:vnby+vnbh, vnbx:vnbx+vnbw] = attacker_nose_bridge_cut
	attacker[vnty:vnty+vnth, vntx:vntx+vntw] = attacker_nose_tip_cut
	

def mouth_cut(attacker, attacker_top_lip, attacker_bottom_lip):
	# turn tuple list into list list
	attacker_top_lip = [[x,y] for x,y in attacker_top_lip]
	attacker_bottom_lip = [[x,y] for x,y in attacker_bottom_lip]

	# get top lip dimensions and modify dimensions to extend a bit past the top lip edges
	a_top_left_top_lip = [attacker_top_lip[0][0]-5, min(attacker_top_lip[2][1], attacker_top_lip[4][1]) - 5]
	a_bottom_right_top_lip = [attacker_top_lip[6][0] + 5, attacker_top_lip[9][1]]

	aul_pts = numpy.array([a_top_left_top_lip, a_bottom_right_top_lip])

	# make a rectangle around the top lip
	aul_rect = cv2.boundingRect(aul_pts)
	aulx,auly,aulw,aulh = aul_rect

	# crop the image to just the size of the rectangle
	attacker_croped_top_lip = attacker[auly:auly+aulh, aulx:aulx+aulw].copy()

	# do the same for the bottom lip
	# get bottom lip dimensions and modify dimensions to extend a bit past the bottom lip edges
	a_top_left_bottom_lip = [attacker_bottom_lip[6][0] - 5, attacker_bottom_lip[9][1]]
	a_bottom_right_bottom_lip = [attacker_bottom_lip[0][0] + 5, attacker_bottom_lip[3][1] + 5]

	abl_pts = numpy.array([a_top_left_bottom_lip, a_bottom_right_bottom_lip])

	# make a rectangle around the bottom lip
	abl_rect = cv2.boundingRect(abl_pts)
	ablx,ably,ablw,ablh = abl_rect

	# crop the image to just the size of the rectangle
	attacker_croped_bottom_lip = attacker[ably:ably+ablh, ablx:ablx+ablw].copy()

	# get a polygon of the attacker's old mouth and cover up with a neutral color
	mouth_poly = numpy.array( [[a_top_left_top_lip, [a_top_left_top_lip[0], a_bottom_right_bottom_lip[1]], a_bottom_right_bottom_lip, [a_bottom_right_bottom_lip[0], a_top_left_top_lip[1]]]], dtype=numpy.int32 )
	color_ref = attacker[a_bottom_right_bottom_lip[1]][a_bottom_right_bottom_lip[0]]
	color_ref = tuple(int(num) for num in color_ref)
	cv2.fillPoly(attacker, mouth_poly, color_ref)

	return (attacker_croped_top_lip, attacker_croped_bottom_lip)
	

def mouth_paste(attacker, victim_top_lip, victim_bottom_lip, attacker_top_lip_cut, attacker_bottom_lip_cut):
	print("[INFO] Adjusting Mouth...")
	# turn tuple list into list list
	victim_top_lip = [[x,y] for x,y in victim_top_lip]
	victim_bottom_lip = [[x,y] for x,y in victim_bottom_lip]

	# get top lip dimensions and modify dimensions to extend a bit past the top lip edges
	v_top_left_top_lip = [victim_top_lip[0][0]-5, min(victim_top_lip[2][1], victim_top_lip[4][1]) - 5]
	v_bottom_right_top_lip = [victim_top_lip[6][0] + 5, victim_top_lip[9][1]]

	vul_pts = numpy.array([v_top_left_top_lip, v_bottom_right_top_lip])

	# find location of victim top lip
	vul_rect = cv2.boundingRect(vul_pts)
	vulx,vuly,vulw,vulh = vul_rect

	# resize attacker top lip to victim top lip size
	attacker_top_lip_cut = cv2.resize(attacker_top_lip_cut, (vulw, vulh))

	# do the same for the bottom lip
	# get bottom lip dimensions and modify dimensions to extend a bit past the bottom lip edges
	v_top_left_bottom_lip = [victim_bottom_lip[6][0] - 5, victim_bottom_lip[9][1]]
	v_bottom_right_bottom_lip = [victim_bottom_lip[0][0] + 5, victim_bottom_lip[3][1] + 5]

	vbl_pts = numpy.array([v_top_left_bottom_lip, v_bottom_right_bottom_lip])

	# find location of victim bottom lip
	vbl_rect = cv2.boundingRect(vbl_pts)
	vblx,vbly,vblw,vblh = vbl_rect

	# resize attacker bottom lip to victim bottom lip size
	attacker_bottom_lip_cut = cv2.resize(attacker_bottom_lip_cut, (vblw, vblh))

	# Copy new nose onto attacker's face
	attacker[vuly:vuly+vulh, vulx:vulx+vulw] = attacker_top_lip_cut
	attacker[vbly:vbly+vblh, vblx:vblx+vblw] = attacker_bottom_lip_cut

"""
def blank_face(image, landmarks, color_ref):
	# get the highest mark on the face
	y = [y for x,y in landmarks["left_eyebrow"]]
	y += [y for x,y in landmarks["right_eyebrow"]]
	
	miny = min(y)

	# get 4 specific points on the face
	tl = list(landmarks["left_eyebrow"][0])
	tl[1] = miny
	tr = list(landmarks["right_eyebrow"][4])
	tr[1] = miny
	bl = list(landmarks["chin"][5])
	br = list(landmarks["chin"][11])

	poly = numpy.array( [[bl, tl, tr, br]], dtype=numpy.int32 )
	color_ref = tuple(int(num) for num in color_ref)
	cv2.fillPoly(image, poly, color_ref)

	#show_image("croped face", image)
"""

# Get the background around the victim's face and transpose it onto the attacker's image
def set_victim_chin(attacker, victim, victim_chin):
	v_chin_points = [[x,y] for x, y in victim_chin]
	v_chin_points += [[image_size, 0], [image_size, image_size], [0, image_size], [0, 0]]
	poly = numpy.array( [v_chin_points], dtype=numpy.int32 )
	#cv2.fillPoly(attacker, poly, (255,255,255))
	
	# Trim nose triangle
	mask_pts = numpy.array(v_chin_points)
	mask_pts = mask_pts - mask_pts.min(axis = 0)
	mask = numpy.zeros(victim.shape[:2], numpy.uint8)
	cv2.drawContours(mask, [mask_pts], -1, (255,255,255), -1, cv2.LINE_AA)
	victim = cv2.bitwise_and(victim, victim, mask=mask)
	
	# Create mask of nose and create inverse mask
	victim_gray = cv2.cvtColor(victim, cv2.COLOR_BGR2GRAY)
	ret, mask = cv2.threshold(victim_gray, 10, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask)
	
	# black-out area of nose in ROI
	attacker_bg = cv2.bitwise_and(attacker, attacker, mask=mask_inv)
	
	# take only region of nose from nose image
	victim_fg = cv2.bitwise_and(victim, victim, mask=mask)
	
	# put nose in roi and modify main image
	attacker = cv2.add(attacker_bg, victim_fg)

	return attacker


def transpose_face(attacker, victim, flag=True):
	# get the dictionary of landmark points from the victim and attacker
	attacker_landmarks = face_recognition.face_landmarks(attacker)[0]
	victim_landmarks = face_recognition.face_landmarks(victim)[0]

	# start grabing features we will need for later
	attacker_nose_bridge_cut, attacker_nose_tip_cut = nose_cut(attacker, attacker_landmarks["nose_bridge"], attacker_landmarks["nose_tip"])
	attacker_mouth_top_cut, attacker_mouth_bottom_cut = mouth_cut(attacker, attacker_landmarks["top_lip"], attacker_landmarks["bottom_lip"])
	attacker_right_eye_cut = eye_cut(attacker, attacker_landmarks["right_eye"])
	attacker_left_eye_cut = eye_cut(attacker, attacker_landmarks["left_eye"])

	# Grab features from the victim as well
	victim_nose_bridge_cut, victim_nose_tip_cut = nose_cut(victim, victim_landmarks["nose_bridge"], victim_landmarks["nose_tip"])
	victim_mouth_top_cut, victim_mouth_bottom_cut = mouth_cut(victim, victim_landmarks["top_lip"], victim_landmarks["bottom_lip"])
	victim_right_eye_cut = eye_cut(victim, victim_landmarks["right_eye"])
	victim_left_eye_cut = eye_cut(victim, victim_landmarks["left_eye"])
	
	chin_blackout(attacker, victim_landmarks["chin"])
	chin_adjust(attacker, victim, attacker_landmarks["chin"], victim_landmarks["chin"])

	attacker = blur_face_2(attacker, 6)

	if flag:
		mouth_paste(attacker, victim_landmarks["top_lip"], victim_landmarks["bottom_lip"], attacker_mouth_top_cut, attacker_mouth_bottom_cut)
		nose_paste(attacker, victim_landmarks["nose_bridge"], victim_landmarks["nose_tip"], attacker_nose_bridge_cut, attacker_nose_tip_cut)	
		eye_paste(attacker, victim_landmarks["right_eye"], attacker_right_eye_cut)
		eye_paste(attacker, victim_landmarks["left_eye"], attacker_left_eye_cut)
	else:
		mouth_paste(attacker, victim_landmarks["top_lip"], victim_landmarks["bottom_lip"], victim_mouth_top_cut, victim_mouth_bottom_cut)
		nose_paste(attacker, victim_landmarks["nose_bridge"], victim_landmarks["nose_tip"], victim_nose_bridge_cut, victim_nose_tip_cut)	
		eye_paste(attacker, victim_landmarks["right_eye"], victim_right_eye_cut)
		eye_paste(attacker, victim_landmarks["left_eye"], victim_left_eye_cut)

	
	#save_image("8", attacker)

	attacker = blur_face_2(attacker, 3)
	chin_blackout(attacker, victim_landmarks["chin"])
	attacker = set_victim_chin(attacker, victim, victim_landmarks["chin"])


	return attacker


def blur_face(spoofed_image, rounds=1):
	print("[INFO] Bluring Spoofed Image...")

	for i in range(rounds):
		spoofed_image = cv2.pyrDown(spoofed_image)

	for i in range(rounds):
		spoofed_image = cv2.pyrUp(spoofed_image)


def blur_face_2(spoofed_image, rounds=6):

	# generate Gaussian pyramid for A
	G = spoofed_image.copy()
	gp = [G]
	for i in range(rounds):
		G = cv2.pyrDown(G)
		gp.append(G)

	# generate Laplacian Pyramid for A
	lp = [gp[rounds-1]]
	for i in range(rounds-1,0,-1):
		GE = cv2.pyrUp(gp[i])
		L = cv2.subtract(gp[i-1],GE)
		lp.append(L)

	# Now add left and right halves of images in each level
	LS = []
	for l in lp:
		LS.append(l)

	# now reconstruct
	ls_ = LS[0]
	for i in range(1,rounds):
		ls_ = cv2.pyrUp(ls_)
		ls_ = cv2.add(ls_, LS[i])

	return ls_

def done():
	#print("Any key to continue")
	#cv2.waitKey()
	cv2.destroyAllWindows()
	exit(0)

def main():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--encodings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-i", "--image1", required=True,
		help="path to first input image (victim)")
	ap.add_argument("-j", "--image2", required=True,
		help="path to second input image (attacker)")
	args = vars(ap.parse_args())

	# load the known faces and embeddings
	print("[INFO] loading encodings...")
	data = pickle.loads(open(args["encodings"], "rb").read())

	print("[INFO] Processing Attacker Image...")
	image1 = process_face(args["image1"])
	print("[INFO] Processing Victim Image...")
	image2 = process_face(args["image2"])

	transpose_face(image1, image2)

	#blur_face(image1)
	#blur_face_2(image1, 9)

	print("[INFO] Complete!!")

if __name__ == "__main__":
	main()