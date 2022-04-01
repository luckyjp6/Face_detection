from feature import RectangleRegion, HaarFeature
from myclassifier import Classifier
import utils
import numpy as np
import math
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle

class Idea:
    def __init__(self):
        self.clfs = []

    def train(self, dataset):
        """
        Trains the Viola Jones classifier on a set of images.
          Parameters:
            dataset: A list of tuples. The first element is the numpy 
              array with shape (m, n) representing the image. The second
              element is its classification (1 or 0).
        """
        print("Computing integral images")
        posNum, negNum = 0, 0
        iis, labels = [], []
        for i in range(len(dataset)):
            iis.append(utils.integralImage(dataset[i][0]))
            labels.append(dataset[i][1])
            if dataset[i][1] == 1:
                posNum += 1
            else:
                negNum += 1
        print("Building features")
        features = self.buildFeatures(iis[0].shape)
        print("Applying features to dataset")
        featureVals = self.applyFeatures(features, iis)
        print("Selecting top3 features")
        indices = SelectPercentile(f_classif, percentile=10).fit(featureVals.T, labels).get_support(indices=True)
        featureVals = featureVals[indices]
        features = features[indices]
        print("Selected %d potential features" % len(featureVals))
            
        self.clfs = self.selectBest(featureVals, iis, labels, features)
            
        accuracy = []
        for x, y in zip(iis, labels):
            correctness = abs(self.classify(x) - y)
            accuracy.append(correctness)
            
        print("accuracy: %f " % (len(accuracy) - sum(accuracy)))
    
    def buildFeatures(self, imageShape):
        """
        Builds the possible features given an image shape.
          Parameters:
            imageShape: A tuple of form (height, width).
          Returns:
            A numpy array of HaarFeature class.
        """
        height = imageShape[0]
        width = imageShape[1]
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(HaarFeature([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(HaarFeature([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(HaarFeature([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(HaarFeature([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(HaarFeature([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)
    
    def applyFeatures(self, features, iis):
        """
        Maps features onto the training dataset.
          Parameters:
            features: A numpy array of HaarFeature class.
            iis: A list of numpy array with shape (m, n) representing the integral images.
          Returns:
            featureVals: A numpy array of shape (len(features), len(dataset)).
              Each row represents the values of a single feature for each training sample.
        """
        featureVals = np.zeros((len(features), len(iis)))
        for j in range(len(features)):
            for i in range(len(iis)):
                featureVals[j, i] = features[j].computeFeature(iis[i])
        return featureVals
    
    def selectBest(self, featureVals, iis, labels, features):
      """
      Finds the appropriate classifiers for each feature.
        Parameters:
          featureVals: A numpy array of shape (len(features), len(dataset)).
            Each row represents the values of a single feature for each training sample.
          iis: A list of numpy array with shape (m, n) representing the integral images.
          labels: A list of integer.
            The ith element is the classification of the ith training sample.
          features: A numpy array of HaarFeature class.
        Returns:
          bestClf: The list of best Classifiers class
      """
      # Begin your code (Part 2)
      """
      bestError : The list of number corresponding to bestClf, 
                  stores the number of wrong classification cases.      
      now, prev[0], prev[1] : the number of wrong classifications
        now : using current standard
        prev[0] : using the standard that is bigger than current standard
        prev[1] : using the standard that is smaller than current standard
      
      train a classifier for each classifier
      adjust standard of current classifier depends on now and prev
      return 11 best classifier(I get the number 11 by testing)
      """     
      bestClf = []
      bestError = []
      
      # fixed the size of bestClf and bestError
      # which is number of the classifiers we will chose
      for i in range(11):
        bestClf.append(Classifier(features[0]))
        bestError.append(2e9)

      for i in range(len(features)):        
        CC = Classifier(features[i])
        
        # initialize prev and now
        prev = [0, 0]
        now = 0        
        for j in range(len(labels)):
          h = featureVals[i][j]
          now = now + 1 if h < CC.standard else now
          prev[1] = prev[1]+1 if h < CC.standard-1 else prev[1]
          prev[0] = prev[0]+1 if j < CC.standard+1 else prev[0]
      
        # add_or_sub : True: add, False: sub
        add_or_sub = (prev[1] > now)
        CC.modify_standard(add_or_sub)

        # set a Maxmum times that standard can be modified
        # in case of oscillating
        COUNT = 0
        while (COUNT < 20):
          COUNT += 1
          temp = now
          CC.modify_standard(add_or_sub)
          
          now = 0
          for j in range(len(labels)):
            h = featureVals[i][j]
            h = 1 if h < CC.standard else 0
            if (h != labels[j]) :
              now += 1
          
          # if now is the best one
          if now <= prev[0] and now <= prev[1]:
            break

          if now > prev[add_or_sub]:
            add_or_sub = not add_or_sub

          prev[add_or_sub] = temp
        
        # find the biggest number in bestError[]
        # and store its index to MAX_BEST
        MAX_BEST = 0
        for k in range(1, len(bestError)):
          if bestError[k] > bestError[MAX_BEST]:
            MAX_BEST = k
        
        # updata the bestClf and bestError
        if now < bestError[MAX_BEST]:
          bestError[MAX_BEST] = now
          bestClf[MAX_BEST] = CC     
        
        # for Clf in bestClf:
        #   print(Clf.standard) 
      return bestClf
    
    

    def classify(self, image):
        """
        Classifies an image by voting, 
        if more than half of the classifier classify the image "Face",
        than the image is classify "Face".
          Parameters:
            image: A numpy array with shape (m, n). The shape (m, n) must be
              the same with the shape of training images.
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = utils.integralImage(image)
        for clf in self.clfs:
            total += clf.classify(ii)
        return 1 if total >= (len(self.clfs)/2) else 0
    
    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)