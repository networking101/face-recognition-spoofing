import cv2
import sys
import os

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

argc = len(sys.argv)
argv = sys.argv

if argc != 3:
    print("Need 2 arguments, try again")
    exit(0)

image1 = cv2.imread(argv[1])
image2 = cv2.imread(argv[2])

h1, w1 = image1.shape[:2]
h2, w2 = image2.shape[:2]

if h1 != h2 and w1 != w2:
    print("Both images need to be the same size")
    print("Image 1: " + str(w1) + " x " + str(h1) + "\tImage 2: " + str(w2) + " x " + str(h2))
    exit(0)

print("Dimensions: " + str(w1) + " x " + str(h1))

same = 0
diff = 0
for i in range(len(image1)):
    for j in range(len(image1[0])):
        t1 = image1[i][j]
        t2 = image2[i][j]
        for k in range(len(t1)):
            if t1[k] != t2[k]:
                diff += 1
                # Debugging, check to see if accurate measurement of different pixels
                image2[i][j] = (0,0,255)
                break
            same += 1

show_image("check_difference", image2)
tot = same + diff
print("Total pixels: " + str(tot) + "\tSame: " + str(same) + "\tDiff: " + str(diff))

print("Same percent: " + str(same/tot*100))
print("Diff percent: " + str(diff/tot*100))