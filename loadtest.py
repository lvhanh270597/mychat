from rasa.nlu import utils
import numpy as np
import os

class Load:

    def __init__(self):
        file_name = "classifier.pkl"
        classifier_file = os.path.join("models/nlu-20200108-095729/nlu", file_name)
        classifier = utils.json_unpickle(classifier_file)
        matrix = np.matrix(classifier.classifier.coef_)
        print(matrix.max())
        print(matrix.min())
        print(classifier.classifier.coef_)


Load()