
class Classifier:
    def __init__(self, feature, standard = 0.0):
        """
          Parameters:
            feature: The HaarFeature class.
            standard: the interger used to classify image
        """
        self.feature = feature
        self.standard = standard
    def __str__(self):
        return "Clf standard = %d, %s" % (self.standard, str(self.feature))

    def modify_standard(self, add_or_sub):
        """
        add_or_sub: True-> add, False-> sub
        """
        if (add_or_sub):
            # self.standard += self.standard/100
            self.standard += 1
        else :
            # self.standard -= self.standard/100
            self.standard -= 1

    def classify(self, x):
        """
        Classifies an integral image based on a feature f 
        and the classifiers threshold and polarity.
          Parameters:
            x: A numpy array with shape (m, n) representing the integral image.
          Returns:
            1 if feature(x) < standard
            0 otherwise
        """
        return 1 if self.feature.computeFeature(x) < self.standard else 0
    