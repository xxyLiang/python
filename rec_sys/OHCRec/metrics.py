import pandas as pd


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
