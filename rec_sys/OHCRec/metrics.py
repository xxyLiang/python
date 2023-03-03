import pandas as pd
from random import sample

class PaperEvaluate:

    @staticmethod
    def evaluate(rating_matrix: pd.DataFrame, y_test: pd.DataFrame, k: int = 50):
        """
        Evaluate performance of the given rating matrix
        :param rating_matrix: [user * thread] user rating matrix
        :param y_test: [user * thread] user engaged thread matrix
        :param k: choose top k threads as prediction for each user
        :return: precision, recall, F1
        """
        r = 0
        hits = 0
        miss = 0
        for uid, row in rating_matrix.iterrows():
            y = y_test.loc[uid]
            row = row.sort_values(ascending=False)
            row = row[row.gt(0)].iloc[:k]
            r += row.shape[0]
            h = y.loc[row.index].sum()
            hits += h
            miss += y.sum() - h

        p = hits / r
        recall = hits / (hits + miss)
        f1 = 2 * p * recall / (p + recall)

        return p, recall, f1

    @staticmethod
    def random_evaluate(rating_matrix: pd.DataFrame, y_test: pd.DataFrame, k: int = 50):
        """
        provide random performance
        :param rating_matrix: [user * thread] user rating matrix. Can be a np.ones() matrix
        :param y_test: [user * thread] user engaged thread matrix
        :param k: RANDOMLY choose k threads as prediction for each user
        :return: precision, recall, F1
        """
        r = 0
        hits = 0
        miss = 0
        for uid, row in rating_matrix.iterrows():
            y = y_test.loc[uid]
            row = row[row.gt(0)]
            if len(row) > k:
                row = row.sample(k)
            r += row.shape[0]
            h = y.loc[row.index].sum()
            hits += h
            miss += y.sum() - h

        p = hits / r
        recall = hits / (hits + miss)
        try:
            f1 = 2 * p * recall / (p + recall)
        except:
            f1 = 0

        return p, recall, f1


class MyEvaluate:
    @staticmethod
    def evaluate(rating_matrix: pd.DataFrame, y_test: pd.DataFrame, engage_matrix, k: int = 10):
        """
        Evaluate performance of the given rating matrix
        :param rating_matrix: [user * thread] user rating matrix
        :param y_test: [user * thread] user engaged thread matrix
        :param k: choose top k threads as prediction for each user
        :return: precision, recall, F1
        """
        pos_rating = rating_matrix * y_test
        neg_rating = rating_matrix * (1 - engage_matrix)

        total = 0
        correct = 0
        for idx, row in pos_rating.iterrows():
            neg_row = neg_rating.loc[idx]
            neg_points = neg_row[neg_row != 0].to_list()
            for t, p in row[row != 0].items():
                neg_points = sample(neg_points, k)
                if max(neg_points) < p:
                    correct += 1
                total += 1

        print("precision = %f" % (correct / total))

