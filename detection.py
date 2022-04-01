import os
import cv2
from cv2 import imshow
import matplotlib.pyplot as plt
from cv2 import INTER_AREA


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
    tt: a list, stores the lines in the .txt file.
    path_list: the full path of the folder "detect".

    in the while loop:
      name: the name of the picture.
      num: how many faces are there in the picture.

      I read the same photo twice,
      img is used to draw rectangles on, and
      gray_img is used to do clssification.

      the format of detectData.txt is:
      index of x, index of y, the width(on x-axis), the lenth(on y-axis).

      part_img: the gray, 19*19 image.

      if the classification is "Face", draw a green rectangle, red otherwise.

      I remove the line from tt whenever I've used the information within,
      this could help me ensure the next line I read is a new one.
    """
    f = open(dataPath, 'r')
    tt = []
    for line in f.readlines():
        tt.append(line)

    path_list = dataPath.split('/')
    sub_path = './' + path_list[0] + '/' + path_list[1] + '/'

    while len(tt) != 0:
        s = tt[0].split(' ')
        name = s[0]
        num = (int)(s[1])
        img = cv2.imread(sub_path + name)
        gray_img = cv2.imread(sub_path + name, cv2.IMREAD_GRAYSCALE)
        tt.remove(tt[0])

        while(num):
            num -= 1
            s = tt[0].split(' ')
            x = (int)(s[0])
            y = (int)(s[1])
            x_range = (int)(s[2])
            y_range = (int)(s[3])

            part_img = gray_img[y:y+y_range, x:x+x_range]
            part_img = cv2.resize(part_img, dsize=(19, 19), interpolation=INTER_AREA)
            
            if clf.classify(part_img) == 1:
                cv2.rectangle(
                    img, (x, y), (x+x_range, y+y_range), (0, 255, 0), 2)
            else:
                cv2.rectangle(
                    img, (x, y), (x+x_range, y+y_range), (0, 0, 255), 2)

            tt.remove(tt[0])

        imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    f.close()
    # raise NotImplementedError("To be implemented")
    # End your code (Part 4)
