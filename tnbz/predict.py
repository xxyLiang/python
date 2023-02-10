import numpy as np
import xgboost
from ensemble_method import DivideEnsembleClassification
from extract_feature import Feature
import pymysql
import train
import pandas as pd
import pickle
import traceback

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()
cursor_update = db.cursor()

pca = pickle.load(open('./model/pca.pickle', 'rb'))
ensemble_SIS = DivideEnsembleClassification(predict_types='SIS')
ensemble_PIS = DivideEnsembleClassification(predict_types='PIS')
ensemble_SES = DivideEnsembleClassification(predict_types='SES')
ensemble_PES = DivideEnsembleClassification(predict_types='PES')
ensemble_COM = DivideEnsembleClassification(predict_types='COM')
clf_SIS = pickle.load(open('./model/predict/SIS.pickle', 'rb'))
clf_PIS = pickle.load(open('./model/predict/PIS.pickle', 'rb'))
clf_SES = pickle.load(open('./model/predict/SES.xgb.pickle', 'rb'))
clf_PES = pickle.load(open('./model/predict/PES.xgb.pickle', 'rb'))
clf_COM = pickle.load(open('./model/predict/COM.xgb.pickle', 'rb'))


def pca_decomposition(df):
    bert_feature_pca = pca.transform(df.iloc[:, :768])
    bert_feature_pca = pd.DataFrame(bert_feature_pca, columns=['feature%d' % i for i in range(100)])
    bert_feature_pca = bert_feature_pca.reset_index(drop=True)
    feature = pd.concat([bert_feature_pca, df.iloc[:, 768:].reset_index(drop=True)], axis=1)
    return feature


def predict(feature):
    sis = ensemble_SIS.transform(feature)
    pis = ensemble_PIS.transform(feature)
    ses = xgboost.DMatrix(ensemble_SES.transform(feature))
    pes = xgboost.DMatrix(ensemble_PES.transform(feature))
    com = xgboost.DMatrix(ensemble_COM.transform(feature))
    predictions = [clf_SIS.predict(sis), clf_PIS.predict(pis), [round(i) for i in clf_SES.predict(ses)],
                   [round(i) for i in clf_PES.predict(pes)], [round(i) for i in clf_COM.predict(com)]]
    return pd.DataFrame(np.transpose(np.array(predictions)), columns=['SIS', 'PIS', 'SES', 'PES', 'COM'])


def save_to_database(pid, predictions):
    rs = pd.concat([predictions, pid], axis=1)
    try:
        for r in rs.values:
            cursor_update.execute("update posts set SIS=%s, PIS=%s, SES=%s, PES=%s, COM=%s where pid=%s", tuple(r))
        db.commit()
    except:
        db.rollback()
        traceback.print_exc()


def run(batch_size=16):
    transformer = Feature()

    total_num = cursor.execute(
        "select pid, is_initiate_post, is_thread_publisher, img_num, content, reply_content "
        "from posts where SIS is null")
    count = 0

    while True:
        batch = cursor.fetchmany(batch_size)
        if len(batch) == 0:
            break

        df = pd.DataFrame(
            train.check_sentences(batch),
            columns=['pid', 'is_init', 'is_reply', 'img_num', 'content', 'reply_content'])

        contents = df.iloc[:, 4]
        reply_contents = df.iloc[:, 5]
        contents_feature = transformer.joint_feature(list(contents))
        replies_feature = transformer.joint_feature(list(reply_contents), bert_feature=False)
        init_reply_img = df.iloc[:, 1:4]

        feature = pd.concat([contents_feature, replies_feature, init_reply_img], axis=1)
        feature = pca_decomposition(feature)

        predictions = predict(feature)       # array (5,)

        save_to_database(df.iloc[:, 0], predictions)

        count += len(batch)
        print('\r[%s%s] %s / %s' %
              (('#' * int(100 * count / total_num)), ' ' * (100 - int(100 * count / total_num)), count, total_num),
              end='')


if __name__ == '__main__':
    run(32)
