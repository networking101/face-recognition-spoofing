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
image_size = 300
# how much to extend the image bounds returned by face capture.  The
# face_recognition.face_locations() function will find a face but
# sometimes features are cut off.  A value is chosen by trial and
# error to include as many features in the picture as possible.
boundary_extension = 50
# path to output images.  Used for debugging
out_path = "test_pic_output/"

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

def process_face(image_file):
	print("[INFO] loading and processing " + image_file + "...")
	image = cv2.imread(image_file)
	image = imutils.resize(image, width=600)
	#show_image("orig process_face " + image_file, image)
	face_boundaries = face_recognition.face_locations(image)[0]
	image = cv2.resize(image[face_boundaries[0]-boundary_extension:face_boundaries[2]+boundary_extension, face_boundaries[3]-boundary_extension:face_boundaries[1]+boundary_extension], (image_size, image_size))
	image = translate_face(image)
	image = rotate_face(image)
	#save_image("new_mark_face", mark_faces(image))
	return image

def chin_overflow(box, color_ref_point):
	delta_x = box['a'][0] - box['v'][0]
	delta_y = box['a'][1] - box['v'][1]

	#print("box: " + str(box))
	#print("color reference point: " + str(color_ref_point))

	if delta_x > 0:
		box['a'][0] += 10
		color_ref_point[0] += 10
	else:
		box['a'][0] -= 10
		color_ref_point[0] -= 10
	if delta_y > 0:
		box['a'][1] += 10
		color_ref_point[1] += 10
	else:
		box['a'][1] -= 10
		color_ref_point[1] -= 10

	#print("new box: " + str(box))
	#print("new color reference point: " + str(color_ref_point))
		
	return (box, color_ref_point)


def chin(attacker, victim, attacker_chin, victim_chin):
	#print(attacker)
	#print(victim)
	#print(attacker_chin)
	#print(victim_chin)

	last_attacker_ref_point = []
	last_attacker_ref_point_orig = []
	last_victim_ref_point = []

	#save_image("attacker_pre", mark_faces(attacker))

	for point in range(len(attacker_chin)):

		box = {'a': list(attacker_chin[point]), 'v': list(victim_chin[point])}

		if point == 0:
			last_attacker_ref_point_orig = [box['a'][0], box['a'][1]]
			box, _ = chin_overflow(box, [0,0])
			last_attacker_ref_point = box['a']
			#print("\n")
			#print(last_attacker_ref_point_orig, last_attacker_ref_point)
			#print("\n")
			last_victim_ref_point = victim_chin[point]
			#print(last_attacker_ref_point)
			#print(last_victim_ref_point)
			continue

		#print("COORDINATES: ", str(box['a'][0]) + "  " + str(last_attacker_ref_point_orig[0]) + "  " + str(box['a'][1]) + "  " + str(last_attacker_ref_point_orig[1]))
		color_ref_point = [int(statistics.mean([attacker_chin[point][0], last_attacker_ref_point_orig[0]])), int(statistics.mean([attacker_chin[point][1], last_attacker_ref_point_orig[1]]))]
		#print("COLOR REFERENCE POINT: " + str(color_ref_point))
		color_ref_point = [int(   (box['a'][0]+last_attacker_ref_point_orig[0])/2   )  ,  int(   (box['a'][1]+last_attacker_ref_point_orig[1])/2   )]
		#print("NEW COLOR REFERENCE POINT: " + str(color_ref_point))

		last_attacker_ref_point_orig = [box['a'][0], box['a'][1]]
		box, color_ref_point = chin_overflow(box, color_ref_point)
		
		#print("Color reference point: " + str(color_ref_point))
		color_ref = attacker[color_ref_point[1]][color_ref_point[0]]
		#print("Color ref: " + str(color_ref))
		#print([last_attacker_ref_point, last_victim_ref_point, box['v'], box['a']])

		poly = numpy.array( [[last_attacker_ref_point, last_victim_ref_point, box['v'], box['a']]], dtype=numpy.int32 )
		color_ref = tuple(int(num) for num in color_ref)
		cv2.fillPoly(attacker, poly, tuple(color_ref))
		#cv2.fillPoly(attacker, poly, (255,255,255))

		last_attacker_ref_point = box['a']
		last_victim_ref_point = victim_chin[point]

		#if point == 16:
	#save_image("attacker_post", attacker)
	#show_image("attacker_post", attacker)

	#done()
	return attacker

def eyes(attacker, victim, attacker_eye, victim_eye):

	eye_bounds = 5

	# turn tuple list into list list
	attacker_eye = [[x,y] for x,y in attacker_eye]
	victim_eye = [[x,y] for x,y in victim_eye]
	#show_image("attacker", attacker)

	# grab a neutral color to cover up old eyes
	color_ref = attacker[int((attacker_eye[4][1] + attacker_eye[5][1])/2) + 20][int((attacker_eye[4][0] + attacker_eye[5][0])/2)]

	are_pts = numpy.array(attacker_eye)
	vre_pts = numpy.array(victim_eye)

	# make a rectangle around all 6 eye points
	are_rect = cv2.boundingRect(are_pts)
	arx,ary,arw,arh = are_rect
	arx -= eye_bounds
	ary -= eye_bounds
	arw += (eye_bounds*2)
	arh += (eye_bounds*2)

	# crop the image to just the size of the rectangle
	attacker_croped_eye = attacker[ary:ary+arh, arx:arx+arw].copy()
	#show_image("attacker_croped_eye", attacker_croped_eye)

	# cover up old eye
	are_cover = []
	are_cover.append([attacker_eye[0][0]-10, attacker_eye[0][1]])
	are_cover.append([attacker_eye[1][0]-5, attacker_eye[1][1]-10])
	are_cover.append([attacker_eye[2][0]+5, attacker_eye[2][1]-10])
	are_cover.append([attacker_eye[3][0]+10, attacker_eye[3][1]])
	are_cover.append([attacker_eye[4][0]+5, attacker_eye[4][1]+10])
	are_cover.append([attacker_eye[5][0]-5, attacker_eye[5][1]+10])
	# adjust size
	poly = numpy.array( [are_cover], dtype=numpy.int32 )
	color_ref = tuple(int(num) for num in color_ref)
	cv2.fillPoly(attacker, poly, color_ref)

	# find location of victim eye
	vre_rect = cv2.boundingRect(vre_pts)
	vrx,vry,vrw,vrh = vre_rect
	vrx -= eye_bounds
	vry -= eye_bounds
	vrw += (eye_bounds*2)
	vrh += (eye_bounds*2)

	# crop attacker eye to victim eye size
	attacker_croped_eye = cv2.resize(attacker_croped_eye, (vrw, vrh))
	#show_image("attacker_croped_eye", attacker_croped_eye)

	# copy new eye onto attackers face
	attacker[vry:vry+vrh, vrx:vrx+vrw] = attacker_croped_eye

	return attacker

def nose(attacker, victim, attacker_nose_bridge, victim_nose_bridge, attacker_nose_tip, victim_nose_tip):
	#print(attacker_nose_bridge)

	# turn tuple list into list list
	attacker_nose_bridge = [[x,y] for x,y in attacker_nose_bridge]
	victim_nose_bridge = [[x,y] for x,y in victim_nose_bridge]
	attacker_nose_tip = [[x,y] for x,y in attacker_nose_tip]
	victim_nose_tip = [[x,y] for x,y in victim_nose_tip]

	# get nose tip width
	a_nose_tip_width = attacker_nose_tip[4][0] - attacker_nose_tip[0][0]
	v_nose_tip_width = victim_nose_tip[4][0] - victim_nose_tip[0][0]
	#print("nose tip width: " + str(a_nose_tip_width))

	# we want to get a triangle cut of the nose bridge to copy into a new location
	# get nose bridge dimensions and modify dimensions to extend a bit past the nose tip in the x direction
	a_top_bridge = attacker_nose_bridge[0]
	a_left_bridge = [attacker_nose_tip[0][0] - int(a_nose_tip_width/2), attacker_nose_bridge[3][1]]
	a_right_bridge = [attacker_nose_tip[4][0] + int(a_nose_tip_width/2), attacker_nose_bridge[3][1]]

	v_top_bridge = victim_nose_bridge[0]
	v_left_bridge = [victim_nose_tip[0][0] - int(v_nose_tip_width/2), victim_nose_bridge[3][1]]
	v_right_bridge = [victim_nose_tip[4][0] + int(v_nose_tip_width/2), victim_nose_bridge[3][1]]

	anb_pts = numpy.array([a_top_bridge, a_left_bridge, a_right_bridge])
	vnb_pts = numpy.array([v_top_bridge, v_left_bridge, v_right_bridge])

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

	# find location of victim nose bridge
	vnb_rect = cv2.boundingRect(vnb_pts)
	vnbx,vnby,vnbw,vnbh = vnb_rect

	# resize attacker nose bridge to victim nose bridge size
	attacker_croped_nose_bridge_mask = cv2.resize(attacker_croped_nose_bridge_mask, (vnbw, vnbh))
	#show_image("attacker_croped_nose_bridge", attacker_croped_nose_bridge_mask)

	# do the same for the nose tip (this time a rectangle)
	# get nose tip dimensions and modify dimensions to extend a bit past the nose tip in the x direction
	a_top_left_tip = [attacker_nose_tip[0][0] - int(a_nose_tip_width/2), attacker_nose_bridge[3][1]]
	a_bottom_right_tip = [attacker_nose_tip[4][0] + int(a_nose_tip_width/2), attacker_nose_tip[3][1]]

	v_top_left_tip = [victim_nose_tip[0][0] - int(v_nose_tip_width/2), victim_nose_bridge[3][1]]
	v_bottom_right_tip = [victim_nose_tip[4][0] + int(v_nose_tip_width/2), victim_nose_tip[3][1]]

	ant_pts = numpy.array([a_top_left_tip, a_bottom_right_tip])
	vnt_pts = numpy.array([v_top_left_tip, v_bottom_right_tip])

	# make a rectangle around the 3 nose bridge points
	ant_rect = cv2.boundingRect(ant_pts)
	antx,anty,antw,anth = ant_rect

	# crop the image to just the size of the rectangle
	attacker_croped_nose_tip = attacker[anty:anty+anth, antx:antx+antw].copy()

	# find location of victim nose tip
	vnt_rect = cv2.boundingRect(vnt_pts)
	vntx,vnty,vntw,vnth = vnt_rect

	# resize attacker nose tip to victim nose tip size
	attacker_croped_nose_tip = cv2.resize(attacker_croped_nose_tip, (vntw, vnth))
	#show_image("attacker_croped_nose_tip", attacker_croped_nose_tip)

	# Copy new nose onto attacker's face
	attacker[vnby:vnby+vnbh, vnbx:vnbx+vnbw] = attacker_croped_nose_bridge_mask
	attacker[vnty:vnty+vnth, vntx:vntx+vntw] = attacker_croped_nose_tip


	#print("dimensions: " + str(a_top_bridge) + " " + str(a_left_bridge) + " " + str(a_right_bridge))
	#show_image("nose", attacker)

	return attacker


def mouth(attacker, victim, attacker_top_lip, victim_top_lip, attacker_bottom_lip, victim_bottom_lip):
	print("top_lip")
	print(attacker_top_lip)
	print("bottom_lip")
	print(attacker_bottom_lip)

	#show_image("mouth", mark_faces(attacker))

	# turn tuple list into list list
	attacker_top_lip = [[x,y] for x,y in attacker_top_lip]
	victim_top_lip = [[x,y] for x,y in victim_top_lip]
	attacker_bottom_lip = [[x,y] for x,y in attacker_bottom_lip]
	victim_bottom_lip = [[x,y] for x,y in victim_bottom_lip]

	# get upper lip dimensions and modify dimensions to extend a bit past the upper lip edges
	a_top_left_upper_lip = [attacker_top_lip[0][0]-5, min(attacker_top_lip[2][1], attacker_top_lip[4][1]) - 5]
	a_bottom_right_upper_lip = [attacker_top_lip[6][0] + 5, attacker_top_lip[9][1]]

	v_top_left_upper_lip = [victim_top_lip[0][0]-5, min(victim_top_lip[2][1], victim_top_lip[4][1]) - 5]
	v_bottom_right_upper_lip = [victim_top_lip[6][0] + 5, victim_top_lip[9][1]]

	aul_pts = numpy.array([a_top_left_upper_lip, a_bottom_right_upper_lip])
	vul_pts = numpy.array([v_top_left_upper_lip, v_bottom_right_upper_lip])

	# make a rectangle around the upper lip
	aul_rect = cv2.boundingRect(aul_pts)
	aulx,auly,aulw,aulh = aul_rect

	# crop the image to just the size of the rectangle
	attacker_croped_upper_lip = attacker[auly:auly+aulh, aulx:aulx+aulw].copy()

	# find location of victim upper lip
	vul_rect = cv2.boundingRect(vul_pts)
	vulx,vuly,vulw,vulh = vul_rect

	# resize attacker upper lip to victim upper lip size
	attacker_croped_upper_lip = cv2.resize(attacker_croped_upper_lip, (vulw, vulh))
	#show_image("attacker_croped_upper_lip", attacker_croped_upper_lip)

	# do the same for the bottom lip
	# get bottom lip dimensions and modify dimensions to extend a bit past the bottom lip edges
	a_top_left_bottom_lip = [attacker_bottom_lip[6][0] - 5, attacker_bottom_lip[9][1]]
	a_bottom_right_bottom_lip = [attacker_bottom_lip[0][0] + 5, attacker_bottom_lip[3][1] + 5]

	v_top_left_bottom_lip = [victim_bottom_lip[6][0] - 5, victim_bottom_lip[9][1]]
	v_bottom_right_bottom_lip = [victim_bottom_lip[0][0] + 5, victim_bottom_lip[3][1] + 5]

	abl_pts = numpy.array([a_top_left_bottom_lip, a_bottom_right_bottom_lip])
	vbl_pts = numpy.array([v_top_left_bottom_lip, v_bottom_right_bottom_lip])

	# make a rectangle around the bottom lip
	abl_rect = cv2.boundingRect(abl_pts)
	ablx,ably,ablw,ablh = abl_rect

	# crop the image to just the size of the rectangle
	attacker_croped_bottom_lip = attacker[ably:ably+ablh, ablx:ablx+ablw].copy()

	# find location of victim bottom lip
	vbl_rect = cv2.boundingRect(vbl_pts)
	vblx,vbly,vblw,vblh = vbl_rect

	# resize attacker bottom lip to victim bottom lip size
	attacker_croped_bottom_lip = cv2.resize(attacker_croped_bottom_lip, (vblw, vblh))
	show_image("attacker_croped_bottom_lip", attacker_croped_bottom_lip)

	# Copy new nose onto attacker's face
	attacker[vuly:vuly+vulh, vulx:vulx+vulw] = attacker_croped_upper_lip
	attacker[vbly:vbly+vblh, vblx:vblx+vblw] = attacker_croped_bottom_lip


	#print("dimensions: " + str(a_top_bridge) + " " + str(a_left_bridge) + " " + str(a_right_bridge))
	#show_image("nose", attacker)

	return attacker


def transpose_face(attacker, victim):
	#save_image("attacker", mark_faces(attacker))
	#save_image("victim", mark_faces(victim))
	attacker_landmarks = face_recognition.face_landmarks(attacker)[0]
	victim_landmarks = face_recognition.face_landmarks(victim)[0]
	print("Attacker")
	print(attacker_landmarks)
	print("Victim")
	print(victim_landmarks)

	#attacker = chin(attacker, victim, attacker_landmarks["chin"], victim_landmarks["chin"])
	#save_image("features1", attacker)
	#attacker = nose(attacker, victim, attacker_landmarks["nose_bridge"], victim_landmarks["nose_bridge"], attacker_landmarks["nose_tip"], victim_landmarks["nose_tip"])
	#save_image("features2", attacker)
	#attacker = eyes(attacker, victim, attacker_landmarks["right_eye"], victim_landmarks["right_eye"])
	#save_image("features3", attacker)
	#attacker = eyes(attacker, victim, attacker_landmarks["left_eye"], victim_landmarks["left_eye"])
	#save_image("features4", attacker)
	attacker = mouth(attacker, victim, attacker_landmarks["top_lip"], victim_landmarks["top_lip"], attacker_landmarks["bottom_lip"], victim_landmarks["bottom_lip"])
	save_image("mouth", attacker)

	# resize to blur the adjustments
	#cv2.resize(attacker, (150, 150))

	#show_image("attacker", attacker)


def done():
	#print("Any key to continue")
	#cv2.waitKey()
	cv2.destroyAllWindows()
	exit(0)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image1", required=True,
	help="path to first input image (victim)")
ap.add_argument("-j", "--image2", required=True,
	help="path to second input image (attacker)")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

image1 = process_face(args["image1"])
rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2 = process_face(args["image2"])
rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
#save_image("orig 1", image1)
#save_image("orig 2", image2)

transpose_face(image1, image2)

done()

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] generating embeddings for face 1...")
boxes1 = face_recognition.face_locations(rgb1,
	model=args["detection_method"])
encodings1 = face_recognition.face_encodings(rgb1, boxes1)

# same for second image
print("[INFO] generating embeddings for face 2...")
boxes2 = face_recognition.face_locations(rgb2,
	model=args["detection_method"])
encodings2 = face_recognition.face_encodings(rgb2, boxes2)

# run the face_recognition.compare_faces() function to match faces
print("[INFO] comparing faces...")
for encoding2 in encodings2:
	match = face_recognition.compare_faces(encodings1, encoding2, .6)
print(match)

"""

# initialize the list of names for each face detected
names = []

# loop over the facial embeddings
for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	matches = face_recognition.compare_faces(data["encodings"],
		encoding)
	name = "Unknown"

	# check to see if we have found a match
	if True in matches:
		# find the indexes of all matched faces then initialize a
		# dictionary to count the total number of times each face
		# was matched
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# loop over the matched indexes and maintain a count for
		# each recognized face face
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1

		# determine the recognized face with the largest number of
		# votes (note: in the event of an unlikely tie Python will
		# select first entry in the dictionary)
		name = max(counts, key=counts.get)
	
	# update the list of names
	names.append(name)

# loop over the recognized faces
for ((top, right, bottom, left), name) in zip(boxes, names):
	# draw the predicted face name on the image
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (0, 255, 0), 2)

# show the output image
cv2.imshow("Image1", image1)
"""
done()



def face_contours(attacker, victim, attacker_chin, victim_chin):
	print(attacker_chin)
	print(victim_chin)

	last_attacker_ref_point = []
	last_victim_ref_point = []

	#save_image("attacker_pre", mark_faces(attacker))

	for point in range(len(attacker_chin)):

		if point == 0:
			last_attacker_ref_point = attacker_chin[point]
			last_victim_ref_point = victim_chin[point]
			continue
		
		attacker_line = []
		ls = LineString([tuple(last_attacker_ref_point), tuple(attacker_chin[point])])
		for f in range(0, int(math.ceil(ls.length)) + 1):
			p = ls.interpolate(f).coords[0]
			pr = map(round, p)
			attacker_line.append(list(pr))

		victim_line = []
		ls = LineString([tuple(last_victim_ref_point), tuple(victim_chin[point])])
		for f in range(0, int(math.ceil(ls.length)) + 1):
			p = ls.interpolate(f).coords[0]
			pr = map(round, p)
			victim_line.append(list(pr))

		print("round: " + str(point))
		print(len(victim_line))
		print(len(attacker_line))
		#for i in range(len(attacker_line))

		#done()

		last_attacker_ref_point = attacker_chin[point]
		last_victim_ref_point = victim_chin[point]
	done()



def final_chin(attacker, victim, attacker_chin, victim_chin):
	print(attacker_chin)
	print(victim_chin)

	last_attacker_ref_point = []
	last_victim_ref_point = []

	#save_image("attacker_pre", mark_faces(attacker))

	for point in range(len(attacker_chin)):

		if point == 0:
			last_attacker_ref_point = attacker_chin[point]
			last_victim_ref_point = victim_chin[point]
			continue
		
		#mask = numpy.zeros((attacker.shape[:-1]), dtype=numpy.uint8)
		box = numpy.array( [[last_attacker_ref_point, last_victim_ref_point, victim_chin[point], attacker_chin[point]]], dtype=numpy.int32 )
		cv2.fillPoly(attacker, box, (255, 255, 255))

		if point == 1:
			#save_image("attacker_post", mark_faces(attacker))

			done()