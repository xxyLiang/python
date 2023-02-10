from numpy import loadtxt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pickle

'''
no_stopwords + tf-idf: [299]	train-error:0.01098	val-error:0.08504, Accuracy:91.50%
[499]	train-error:0.02433	val-error:0.08289  Accuracy:91.71%
[769]	train-error:0.03064	val-error:0.08046  Accuracy:91.92%
recall: 0.9229861930201901
Precision: 0.9181447995030713
F1-score: 0.920559130855996
AUC: 0.9191299017679889
Accuracy:91.92%
confusion matrix:
[[12812,  1186],
[ 1110, 13303]]

with_stopwords+ tf-idf:
[601]	train-error:0.04065	val-error:0.08111
recall: 0.916499
Precision: 0.918282
F1-score: 0.917390
AUC: 0.918025
Accuracy:91.80%
Confusion matrix:
[[12962  1134]
 [ 1161 12743]]
'''

def run(dataset):
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    seed = 7
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    data_train = xgb.DMatrix(X_train, label=y_train)
    data_test = xgb.DMatrix(X_test, label=y_test)

    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        # 'objective': 'multi:softmax',
        # 'num_class': 3,
        'learning_rate': 0.05,
        'gamma': 0.1,
        'max_depth': 6,
        'min_child_weight': 3,
        'alpha': 2,
        'lambda': 1.5,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'random_state': 7,
        'silent': 0,
        'eta': 0.01,
        'tree_method': 'gpu_hist',
        # 'nthread': 1
    }
    watchlist = [(data_train, 'train'), (data_test, 'val')]
    num_rounds = 1000
    model = xgb.train(params, xgb.DMatrix(X_train, y_train), num_rounds, evals=watchlist, early_stopping_rounds=100, verbose_eval=10)
    model.save_model('./result_model/train_result')

    y_pred = model.predict(xgb.DMatrix(X_test))
    predictions = [round(value) for value in y_pred]

    print(X_train.shape, '\n', X_test.shape)
    print("recall: %f" % (recall_score(y_test, predictions)))
    print("Precision: %f" % (precision_score(y_test, predictions)))
    print("F1-score: %f" % (f1_score(y_test, predictions)))
    print("AUC: %f" % (roc_auc_score(y_test, predictions)))
    print("Accuracy:%.2f%%" % (accuracy_score(y_test, predictions) * 100.0))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, predictions))


def score():
    dataset = loadtxt(r'D:\train_data_stopwords.csv', delimiter=',')
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    seed = 7
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    model = xgb.Booster()
    model.load_model('./result_model/train_result')
    y_pred = model.predict(xgb.DMatrix(X_test))
    predictions = [round(value) for value in y_pred]

    print("recall: %f" % (recall_score(y_test, predictions)))
    print("Precision: %f" % (precision_score(y_test, predictions)))
    print("F1-score: %f" % (f1_score(y_test, predictions)))
    print("AUC: %f" % (roc_auc_score(y_test, predictions)))
    print("Accuracy:%.2f%%" % (accuracy_score(y_test, predictions) * 100.0))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, predictions))


if __name__ == '__main__':
    data = loadtxt(r'D:\train_data_stopwords.csv', delimiter=',')
    run(data)
    # score()