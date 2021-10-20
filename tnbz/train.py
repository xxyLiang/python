from ensemble_method import DivideEnsembleClassification
import numpy as np
import os
import pandas as pd
import pymysql
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import pickle
import xgboost
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


def check_sentences(sql_results):
    new_results = []
    for r in sql_results:
        if r[4] == '' or r[4] == ' ':  # 空内容，删掉
            continue
        temp = list(r)  # 转list，以便将过长的句子截短
        if len(temp[4]) > 2000:
            temp[4] = temp[4][:2000]
        if temp[5] is None:
            temp[5] = ''
        new_results.append(temp)
    return new_results


def get_train_vector(batch_size=16):
    # if vector file can not found:
    # load raw train data from database, transform it to vector/features, and store to local disk (DataFrame).
    from extract_feature import Feature
    print("Start to transform train_data to vector.")
    transformer = Feature()
    train_vector = pd.DataFrame()
    db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
    cursor = db.cursor()
    total_num = cursor.execute("select * from `train_table` order by rand()")
    count = 0
    while True:
        batch = cursor.fetchmany(batch_size)
        if len(batch) == 0:
            break
        df = pd.DataFrame(check_sentences(batch), columns=['pid', 'is_init', 'is_reply', 'img_num', 'content',
                                                           'reply_content', 'SIS', 'PIS', 'SES', 'PES', 'COM'])
        contents = df.iloc[:, 4]
        reply_contents = df.iloc[:, 5]
        feature = transformer.joint_feature(list(contents))
        reply_feature = transformer.joint_feature(list(reply_contents), bert_feature=False)
        init_reply_img = df.iloc[:, 1:4]
        tags = df.iloc[:, -5:]
        vector = pd.concat([feature, reply_feature, init_reply_img, tags], axis=1)
        train_vector = train_vector.append(vector)
        count += len(batch)
        print('\r[%s%s] %s / %s' %
              (('#' * int(50 * count / total_num)), ' ' * (50 - int(50 * count / total_num)), count, total_num),
              end='')
    train_vector = train_vector.reset_index(drop=True)
    train_vector.to_pickle('./model/train_vector.df')
    db.close()
    print("\nTransform Vector Complete.")
    return train_vector


def pca_transform(df, retrain=False):
    dim = 100
    retrain = retrain or not os.path.exists('./model/pca.pickle')
    if retrain:
        pca = PCA(n_components=dim)
        bert_feature_pca = pca.fit_transform(df.iloc[:, :768])
    else:
        pca = pickle.load(open('./model/pca.pickle', 'rb'))
        bert_feature_pca = pca.transform(df.iloc[:, :768])
    bert_feature_pca = pd.DataFrame(bert_feature_pca, columns=['feature%d' % i for i in range(dim)])
    bert_feature_pca = bert_feature_pca.reset_index(drop=True)
    feature = pd.concat([bert_feature_pca, df.iloc[:, 768:-5].reset_index(drop=True)], axis=1)
    if retrain:
        pickle.dump(pca, open('./model/pca.pickle', 'wb'))
    return feature


def performance(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    # print("recall: %f" % (recall_score(y_test, y_pred)))
    # print("Precision: %f" % (precision_score(y_test, y_pred)))
    # print("F1-score: %f" % (f1_score(y_test, y_pred)))
    print("Accuracy:%.3f" % (accuracy_score(y_test, y_pred)))
    print("AUC: %.3f" % (roc_auc_score(y_test, y_pred)))
    print("Confusion matrix:\n")
    print(confusion_matrix(y_test, y_pred))


def search_params(x, y):
    # train_data = xgboost.DMatrix(x, label=y)
    other_params = {
        'n_estimators': 200,
        'booster': 'gbtree',
        'learning_rate': 0.05,
        'gamma': 0.03,
        'max_depth': 4,
        'min_child_weight': 2,
        'alpha': 3,
        'lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'eta': 0.01,
        'eval_metric': ['logloss', 'auc'],
    }
    cv_params = {'n_estimators': [50, 100, 150, 200], 'learning_rate': [0.1, 0.2, 0.5, 1]}
    model = AdaBoostClassifier()
    gs = GridSearchCV(model, cv_params, verbose=2, cv=5, n_jobs=-1, scoring='roc_auc')
    gs.fit(x, y)
    print(gs.cv_results_['mean_test_score'])
    print(gs.cv_results_['rank_test_score'])
    print("参数最佳取值：", gs.best_params_)
    print("最佳模型得分：", gs.best_score_)


def xgb(x_train, x_test, y_train, y_test, predict_types, save_model=False):
    data_train = xgboost.DMatrix(x_train, label=y_train)
    data_test = xgboost.DMatrix(x_test)
    params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'learning_rate': 0.05,
        'gamma': 0.03,
        'max_depth': 5,
        'min_child_weight': 3,
        'alpha': 3,
        'lambda': 10,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'eta': 0.01,
        'random_state': 7,
        # 'silent': 0,
    }
    cv_res = xgboost.cv(params, data_train, num_boost_round=500, verbose_eval=10, nfold=10,
                        metrics='auc', early_stopping_rounds=20, show_stdv=True)
    model = xgboost.train(params, data_train, cv_res.shape[0])
    predictions = model.predict(data_test)
    y_pred = [round(i) for i in predictions]
    performance(y_test, y_pred)
    if save_model:
        pickle.dump(model, open('./model/predict/%s.xgb.pickle' % predict_types, 'wb'))


def run_model(X_train, X_test, y_train, y_test, predict_types, save_model=False):
    # clf = svm.SVC(class_weight='balanced', C=10, gamma=0.001)
    # clf = GaussianNB()
    clf = LogisticRegression(class_weight='balanced')
    # clf = GradientBoostingClassifier(n_estimators=200, max_depth=6, subsample=0.9)
    # clf = AdaBoostClassifier(learning_rate=0.1, n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    performance(y_test, y_pred)
    if save_model:
        pickle.dump(clf, open('./model/predict/%s.pickle' % predict_types, 'wb'))


def run(force_trans=0, save_model=False):
    if os.path.exists('./model/train_vector.df') and not force_trans:
        df = pd.read_pickle('./model/train_vector.df')
    else:
        df = get_train_vector()
    smo = SMOTE(random_state=7, sampling_strategy=0.4)
    feature = pca_transform(df)
    for types in ['PIS']:
        tag = df[types]
        X_train, X_test, y_train, y_test = train_test_split(feature, tag, test_size=0.25, random_state=8)
        ensemble = DivideEnsembleClassification(predict_types=types, retrain=save_model)
        X_train, X_test = ensemble.fit_transform(X_train, y_train, X_test)
        if types in ['SES', 'PES']:
            X_train, y_train = smo.fit_resample(X_train, y_train)
        print("********%s********" % types)
        # xgb(X_train, X_test, y_train, y_test, types, save_model)
        run_model(X_train, X_test, y_train, y_test, types, save_model)
    # tag = df['PIS']
    # ensemble = DivideEnsembleClassification()
    # feature, _ = ensemble.fit_transform(feature, tag, feature)
    # search_params(feature, tag)

    # clf = svm.SVC(class_weight={0: 0.2, 1: 0.8})


if __name__ == '__main__':
    run()
