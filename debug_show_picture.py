import cv2
import sys
import os

argc = len(sys.argv)
argv = sys.argv

for filename in argv[1:]:
    if os.path.exists(filename):
        print("Press any key to continue")
        image = cv2.imread(filename)
        cv2.imshow(filename, image)
        cv2.waitKey()
    else:
        print("No file\t\t" + filename)

cv2.destroyAllWindows()