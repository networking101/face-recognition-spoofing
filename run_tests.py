# test_uniqueness.py
# This is test code to compare the similarity of the 128 embeddings
# between different pictures.  Faces are reduced to a 300x300 pixel
# image of only the face and sent through the face_recognition.compare_faces()
# function in face_recognition/api.py  I edited the compare_faces()
# function to print out successful authentication.

# example usage:
# python test_uniqueness.py --encodings encodings.pickle --image1 michael_pics/alec1.jpg --image2 michael_pics/adrian1.jpg

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import cv2
import math
import numpy
import copy
import os
import time
import json
import copy

import face_adjustment

# size of the picture frame (image_size x image_size)
image_size = 512
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
	cv2.destroyAllWindows()

def save_image(name, image):
	print("[INFO] image saved to " + out_path + name)
	cv2.imwrite(out_path + name + ".jpg", image)

def done():
	print("Any key to continue")
	cv2.waitKey()
	cv2.destroyAllWindows()
	exit(0)

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

	h, w = image.shape[:2]
	cv2.circle(image, (int(w/2), int(h/2)), 2, (255,0,0), 2)
	
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


def blur_face(spoofed_image, rounds=1):
	print("[INFO] Bluring Spoofed Image...")

	for i in range(rounds):
		spoofed_image = cv2.pyrDown(spoofed_image)

	for i in range(rounds):
		spoofed_image = cv2.pyrUp(spoofed_image)

	return spoofed_image


def chin_blackout(image, chin_points):
	v_chin_points = [[x,y] for x, y in chin_points]
	v_chin_points += [[image_size, 0], [image_size, image_size], [0, image_size], [0, 0]]
	poly = numpy.array( [v_chin_points], dtype=numpy.int32 )
	cv2.fillPoly(image, poly, (0, 0, 0))

start = time.time()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-a", "--attacker", required=True,
	help="path to attacker image")
ap.add_argument("-v", "--victims", required=True,
	help="path to directory of victim images")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the attacker image and convert it from BGR to RGB
attacker = cv2.imread(args["attacker"])
attacker = imutils.resize(attacker, width=600)
small_faces = face_recognition.face_locations(attacker)
for each in small_faces:
	# This increases the bounds of the found face images, sometimes features
	# are cut off when face_recognition.face_locations() is run
	attacker_face = attacker[each[0]-boundary_extension:each[2]+boundary_extension, each[3]-boundary_extension:each[1]+boundary_extension]
	try:
		attacker_face = cv2.resize(attacker_face, (image_size, image_size))
	except:
		attacker_face = cv2.resize(image1, (image_size, image_size))
	attacker_face = translate_face(attacker_face)
	attacker_face = rotate_face(attacker_face)
	break

# cycle through images in test directory
if not (os.path.isdir(args["victims"]) and os.path.exists(args["victims"])):
	print("[ERROR] Bad path to victims directory")
	done()
if args["victims"][:-1] != '/':
	args["victims"] = args["victims"] + '/'
victims = os.listdir(args["victims"]) 
victims.sort()

# save results to an array
results_val = []
results = []
# loop though all victims
for victim in victims:
	# load the victim image and convert it from BGR to RGB
	victim_pic = cv2.imread(args["victims"] + victim)
	victim_pic = imutils.resize(victim_pic, width=600)
	small_faces = face_recognition.face_locations(victim_pic)
	for each in small_faces:
	# if the original file is already reduced, the boundary_extension code will break it
		victim_face = victim_pic[each[0]-boundary_extension:each[2]+boundary_extension, each[3]-boundary_extension:each[1]+boundary_extension]
		try:
			victim_face = cv2.resize(victim_face, (image_size, image_size))
		except:
			victim_face = cv2.resize(image2, (image_size, image_size))
		victim_face = translate_face(victim_face)
		victim_face = rotate_face(victim_face)
		break
	rgb2 = cv2.cvtColor(victim_face, cv2.COLOR_BGR2RGB)

	attacker_face_copy = copy.deepcopy(attacker_face)
	spoofed_attacker_face = face_adjustment.transpose_face(attacker_face_copy, victim_face, False)
	rgb1 = cv2.cvtColor(spoofed_attacker_face, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces for attacker...")
	boxes1 = face_recognition.face_locations(rgb1,
		model=args["detection_method"])
	encodings1 = face_recognition.face_encodings(rgb1, boxes1)

	# detect the (x, y)-coordinates of the bounding boxes corresponding
	# to each face in the input image, then compute the facial embeddings
	# for each face
	print("[INFO] recognizing faces for victim " + victim + "...")
	boxes2 = face_recognition.face_locations(rgb2,
		model=args["detection_method"])
	encodings2 = face_recognition.face_encodings(rgb2, boxes2)

	for encoding2 in encodings2:
		match, match_val = face_recognition.compare_faces(encodings1, encoding2, .6)
		break
	results.append(match[0])
	results_val.append(match_val)
	print(results)
	print(results_val)
	

# save results to a file
export_results = {}
export_results_val = {}

# if file exists, append to that
filename = "face_recognition_results"
if os.path.exists(filename + "_val.json"):
	with open(filename + "_val.json") as f:
		export_results_val = json.load(f)

attacker_name = args["attacker"].split('/')[-1].split('.')[0][:-1]
export_results[attacker_name] = results
export_results_val[attacker_name] = results_val

with open(filename + "_val.json", 'w') as f:
	json.dump(export_results_val, f)

end = time.time()
print("Elapsed time: " + str(end-start))
done()

cv2.destroyAllWindows()