from copy import deepcopy
from pandas.core.generic import NDFrame
import pandas as pd
import numpy as np
from sklearn import svm
import pickle
import os


class DivideEnsembleClassification:

    def __init__(self, predict_types, clf=svm.SVC(), retrain=False):
        self.types = predict_types
        self.clf = clf
        self.retrain = retrain or not os.path.exists('./model/DivideEnsemble/Classifiers_%s.pickle' % self.types)
        if self.retrain:
            self.classifier_list = []
        else:
            self.classifier_list = pickle.load(open('./model/DivideEnsemble/Classifiers_%s.pickle' % self.types, 'rb'))

    def fit(self, x: NDFrame, y: NDFrame, sample_weight=None):
        X_train = x.reset_index(drop=True)
        y_train = y.reset_index(drop=True)
        df = pd.concat([X_train, y_train], axis=1)

        small_class = df[df.iloc[:, -1] == 1]
        big_class = df[df.iloc[:, -1] != 1]

        ratio = big_class.shape[0] / small_class.shape[0]

        if ratio < 1:
            small_class, big_class = big_class, small_class
            ratio = 1 / ratio

        ratio = int(ratio) if int(ratio) % 2 == 1 else int(ratio + 1)  # ratio取奇数整数
        batch_size = int(big_class.shape[0] / ratio + 1)

        self.classifier_list.clear()

        # 训练K个子分类器，正负样本约为1：1
        for i in range(ratio):
            c = deepcopy(self.clf)

            big_class_batch = big_class.iloc[batch_size * i: batch_size * (i + 1)]
            classifier_batch = pd.concat([big_class_batch, small_class])
            classifier_batch = classifier_batch.sample(frac=1).reset_index(drop=True)  # shuffle

            c.fit(classifier_batch.iloc[:, :-1], classifier_batch.iloc[:, -1], sample_weight)

            self.classifier_list.append(c)

        if self.retrain:
            pickle.dump(self.classifier_list, open('./model/DivideEnsemble/Classifiers_%s.pickle' % self.types, 'wb'))

    def transform(self, x):
        x = x.reset_index(drop=True)
        x_predicted_feature = pd.DataFrame(np.transpose(self.predict(x)))
        x_concat = pd.concat([x, x_predicted_feature], axis=1)
        return x_concat

    def predict(self, x_test):
        """
        :param x_test: Test set Matrix
        :return: Predict Results of K sub-classifier, type=array, shape=(K, len of X_test)
        """
        pred_list = []

        if len(self.classifier_list) == 0:
            raise NotImplementedError("Classifier must be fit() first.")

        for c in self.classifier_list:
            y_pred = c.predict(x_test)
            pred_list.append(y_pred)
        pred_list = np.array(pred_list)

        return pred_list

    def fit_transform(self, x_train, y_train, x_test):
        self.fit(x_train, y_train)
        return self.transform(x_train), self.transform(x_test)
