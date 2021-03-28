from numpy import loadtxt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pickle

dataset = loadtxt(r'D:\train_data_stopwords.csv', delimiter=',')

X = dataset[:, 0:-1]
Y = dataset[:, -1]

# seed = 7
# test_size = 0.2

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
other_params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'gamma': 0.1,
    'max_depth': 6,
    'min_child_weight': 3,
    # 'lambda': 1.5,
    # 'alpha': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'random_state': 7,
    'silent': 0,
    'eta': 0.01,
    'tree_method': 'gpu_hist',
    # 'n_estimators': 700
}
cv_params = {'lambda': [0, 0.5, 1, 2], 'alpha': [0, 0.5, 1, 2]}
# cv_params = {'max-depth': [6, 7]}

model = xgb.XGBClassifier(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=3)
optimized_GBM.fit(X, Y)
with open("./result_model/params_result", 'wb') as f:
    pickle.dump(optimized_GBM, f)

evalute_result = optimized_GBM.cv_results_

print('每轮迭代运行结果:{0}'.format(evalute_result))
print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

