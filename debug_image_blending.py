import cv2
import numpy as np,sys
import imutils

A = cv2.imread('test_pic_output/a.jpg')
B = cv2.imread('test_pic_output/b.jpg')

A = imutils.resize(A, width=1024)
B = imutils.resize(B, width=1024)

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

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:(int(cols/2))], lb[:,(int(cols/2)):]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
show_image("test", ls_)
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# image with direct connecting each half
real = np.hstack((A[:,:(int(cols/2))],B[:,(int(cols/2)):]))

cv2.imwrite('test_pic_output/Pyramid_blending.jpg',ls_)
cv2.imwrite('test_pic_output/Direct_blending.jpg',real)