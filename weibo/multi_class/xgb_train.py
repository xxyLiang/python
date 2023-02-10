from numpy import loadtxt
import xgboost as xgb
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

'''
order by length(content) desc
[272]	train-merror:0.00438	val-merror:0.17404
Accuracy:82.51%
Confusion matrix:
[[1135   78  161   81]
 [  54 1404   25   24]
 [ 114   49 1235  124]
 [ 136   54  148 1171]]
'''


def run():
    dataset = loadtxt('D:/train_data_multi.csv', delimiter=',')
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    seed = 7
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    data_train = xgb.DMatrix(X_train, label=y_train)
    data_test = xgb.DMatrix(X_test, label=y_test)


    params = {
        'booster': 'gbtree',
        # 'objective': 'binary:logistic',
        'objective': 'multi:softmax',
        'num_class': 4,
        'learning_rate': 0.1,
        'gamma': 0.2,
        'max_depth': 12,
        'min_child_weight': 2,
        'alpha': 2.5,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'random_state': 7,
        'silent': 0,
        'eta': 0.01,
        # 'nthread': 2,
        'tree_method': 'gpu_hist',
    }
    watchlist = [(data_train, 'train'), (data_test, 'val')]
    num_rounds = 500
    model = xgb.train(params, xgb.DMatrix(X_train, y_train), num_rounds, evals=watchlist, early_stopping_rounds=50, verbose_eval=10)
    # model.save_model('train_result')

    # cv_res = xgb.cv(params, data_train, num_boost_round=50, early_stopping_rounds=50, metrics='merror',
    #                 nfold=10, callbacks=[xgb.callback.print_evaluation(show_stdv=False), xgb.callback.early_stop(5)])
    # print(cv_res)
    #
    # model = xgb.train(params, data_train, num_boost_round=cv_res.shape[0])

    y_pred = model.predict(xgb.DMatrix(X_test))
    predictions = [round(value) for value in y_pred]

    print("Accuracy:%.2f%%" % (accuracy_score(y_test, predictions) * 100.0))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, predictions))


def score():
    import matplotlib.pyplot as plt

    dataset = loadtxt('D:/train_data_multi.csv', delimiter=',')
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]

    seed = 7
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    model = xgb.Booster()
    model.load_model('train_result')
    y_pred = model.predict(xgb.DMatrix(X_test))
    predictions = [round(value) for value in y_pred]

    print("Accuracy:%.2f%%" % (accuracy_score(y_test, predictions) * 100.0))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, predictions))
    plt.annotate()


# if __name__ == '__main__':
#     data = loadtxt(r'D:\train_data_stopwords.csv', delimiter=',')
#     run(data)
#     # score()
run()
# score()

