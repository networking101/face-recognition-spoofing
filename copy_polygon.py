import cv2
import numpy as np

# path to output images.  Used for debugging
out_path = "test_pic_output/"

def show_image(name, image):
	print("Press any key to continue")
	cv2.imshow(name, image)
	cv2.waitKey()
	#cv2.destroyAllWindows()


# load images
img1 = cv2.imread("test_pic_output/bbbbb.jpg")
img2 = cv2.imread("test_pic_output/aaaaa.jpg")

# Create ROI
rows,cols =  img2.shape[:2]
roi = img1[0:rows, 0:cols]

# Trim nose triangle
mask_pts = np.array([[48,0], [0, 65], [96, 65]])
mask = np.zeros(img2.shape[:2], np.uint8)
cv2.drawContours(mask, [mask_pts], -1, (255,255,255), -1, cv2.LINE_AA)
img2 = cv2.bitwise_and(img2, img2, mask=mask)
#show_image("img2", img2)

# Create mask of nose and create inverse mask
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# black-out area of nose in ROI
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# take only region of nose from nose image
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)
show_image("img2_fg", img2_fg)

# put nose in roi and modify main image
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst




"""

# load two images
src1 = cv2.imread("test_pic_output/aaaaa.jpg")
src2 = cv2.imread("test_pic_output/bbbbb.jpg")

# create mask template
src1_mask = src1.copy()
src1_mask = cv2.cvtColor(src1_mask, cv2.COLOR_BGR2GRAY)
#show_image("mask", src1_mask)
src1_mask.fill(0)

# define polygon around region
poly = np.array([ [48,0], [0, 65], [96, 65] ], np.int32)

# fill polygon in mask
_ = cv2.fillPoly(src1_mask, [poly], 255)
#show_image("mask", src1_mask)

#ret, src1_mask = cv2.threshold(src1_mask, 10, 255, cv2.THRESH_BINARY)

# create region of interest
rows,cols =  src1.shape[:2]
roi = src2[20:rows+20, 20:cols+20]
mask = src1_mask[np.min(poly[:,1]):np.max(poly[:,1]),np.min(poly[:,0]):np.max(poly[:,0])]

mask_inv = cv2.bitwise_not(mask)
ret, mask_inv = cv2.threshold(mask_inv, 127, 255, cv2.THRESH_BINARY)
show_image("mask_inv", mask_inv)
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
src1_cut = src1[np.min(poly[:,1]):np.max(poly[:,1]),np.min(poly[:,0]):np.max(poly[:,0])]

img2_fg = cv2.bitwise_and(src1_cut, src1_cut, mask=mask)

# put nose in ROI and modify the main image
dst = cv2.add(img1_bg, img2_fg)
src2_final = src2.copy()
src2_final[np.min(poly[:,1]):np.max(poly[:,1]),np.min(poly[:,0]):np.max(poly[:,0])] = dst

plt.imshow(cv2.cvtColor(src2_final, cv2.COLOR_BGR2RGB))
"""



"""

# make mask of nose triangle and trim out unnecessary face features
mask_pts = anb_pts - anb_pts.min(axis = 0)
mask = numpy.zeros(attacker_croped_nose_bridge.shape[:2], numpy.uint8)
cv2.drawContours(mask, [mask_pts], -1, (255,255,255), -1, cv2.LINE_AA)
attacker_croped_nose_bridge_mask = cv2.bitwise_and(attacker_croped_nose_bridge, attacker_croped_nose_bridge, mask=mask)
# still deciding if we need a white or black mask
bg = numpy.ones_like(attacker_croped_nose_bridge, numpy.uint8)*255
cv2.bitwise_not(bg, bg, mask=mask)
attacker_croped_nose_bridge_mask += bg
"""