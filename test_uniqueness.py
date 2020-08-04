# USAGE
# python recognize_faces_image.py --encodings encodings.pickle --image examples/example_01.png 

# import the necessary packages
import face_recognition
import argparse
import imutils
import pickle
import cv2
import math
import numpy

image_size = 300
boundary_extension = 50

out_path = "test_pic_output/"

def show_image(name, image):
	print("Press any key to continue")
	cv2.imshow(name, image)
	cv2.waitKey()
	#cv2.destroyAllWindows()

# This function marks the locations of the face for face identification.
# The red dots are part of the 68 point ID and the blue point is the
# center of the image
def mark_faces(image):
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



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image1", required=True,
	help="path to first input image")
ap.add_argument("-j", "--image2", required=True,
	help="path to second input image")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# load the input image and convert it from BGR to RGB
images1 = []
image1 = cv2.imread(args["image1"])
image1 = imutils.resize(image1, width=600)
#cv2.imshow("Image1", image1)
small_faces = face_recognition.face_locations(image1)
for each in small_faces:
	# This increases the bounds of the found face images, sometimes features
	# are cut off when face_recognition.face_locations() is run
	images1.append(cv2.resize(image1[each[0]-boundary_extension:each[2]+boundary_extension, each[3]-boundary_extension:each[1]+boundary_extension], (image_size, image_size)))
# Debug printing, do not delete
"""
count = 0
for each in images1:
	cv2.imwrite(out_path + "face1_" + str(count) + ".jpg", each)
	show_image(image)
	count += 1
"""
rgb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# load the second image
images2 = []
image2 = cv2.imread(args["image2"])
image2 = imutils.resize(image2, width=600)
#cv2.imshow("Image2", image2)
small_faces = face_recognition.face_locations(image2)
for each in small_faces:
	temp = cv2.resize(image2[each[0]-boundary_extension:each[2]+boundary_extension, each[3]-boundary_extension:each[1]+boundary_extension], (image_size, image_size))
	temp = translate_face(temp)
	temp = rotate_face(temp)
	images2.append(temp)
	
count = 0
for images in images2:
	#show_image(images)
	#cv2.imwrite(out_path + "face2_" + str(count) + ".jpg", images)
	# These are my notes for debugging, do not delete
	"""
	# first dimension is y axis
	print(len(images))
	# second dimension is x axis
	print(len(images[0]))
	# third dimension is bgr
	print(len(images[0][0]))
	print(images)
	"""
	count += 1
rgb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

show_image("temp", mark_faces(images2[0]))

#exit(0)

# detect the (x, y)-coordinates of the bounding boxes corresponding
# to each face in the input image, then compute the facial embeddings
# for each face
print("[INFO] recognizing faces for image 1...")
boxes1 = face_recognition.face_locations(rgb1,
	model=args["detection_method"])
encodings1 = face_recognition.face_encodings(rgb1, boxes1)

# same for second image
print("[INFO] recognizing faces for image 2...")
boxes2 = face_recognition.face_locations(rgb2,
	model=args["detection_method"])
encodings2 = face_recognition.face_encodings(rgb2, boxes2)

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
print("Any key to continue")
cv2.waitKey()
cv2.destroyAllWindows()