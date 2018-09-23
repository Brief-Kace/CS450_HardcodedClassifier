from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np
#import pandas



class Hard_Coded_Model():
    def predict(self, data_test):
        targets = np.array([], dtype=int)
        for i in range(len(data_test)):
            targets = np.append(targets, 0)
        return targets


class HardCodedClassifier():
    def fit(self, data_train, targets_train):
        self.predict = Hard_Coded_Model().predict




def main():
    iris = datasets.load_iris()
    print(iris)
    iris = np.genfromtxt('iris.csv', delimiter=',')
    print(iris)

    data_train, data_test, targets_train, targets_test = train_test_split( iris.data, iris.target, test_size=.3,
                                                                          random_state=56)

    classifier = GaussianNB()
    classifier.fit(data_train, targets_train)
    targets_predicted = classifier.predict( data_test)

    hc_classifier = HardCodedClassifier()
    hc_classifier.fit( data_train, targets_train)
    hc_targets_predicted = hc_classifier.predict( data_test)
    print("Accuracy  Gaussian:", accuracy_score(targets_test, targets_predicted))
    print("Accuracy HardCoded:", accuracy_score(targets_test, hc_targets_predicted))


main()