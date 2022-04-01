import os
import cv2
import numpy as np

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    """
    face_path: the full path of the folder "face".
    list_of_filename: The list of names of the files in the folder "face".

    use cv2.imread() to read the images respectively, 
    and append to dataset, which is the return value.

    do the same thing with the folder "non-face".
    """
    dataset = []

    face_path = os.path.join('.', dataPath, 'face')
    list_of_filename = os.listdir(face_path)

    for name in list_of_filename:
      re = cv2.imread(os.path.join(face_path, name), cv2.IMREAD_GRAYSCALE)
      dataset.append((re, 1))
    

    non_face_path = os.path.join('.', dataPath, 'non-face')
    list_of_filename = os.listdir(non_face_path)

    for name in list_of_filename:
      re = cv2.imread(os.path.join(non_face_path, name), cv2.IMREAD_GRAYSCALE)
      dataset.append((re, 0))

    # raise NotImplementedError("To be implemented")    
    # End your code (Part 1)
    return dataset
