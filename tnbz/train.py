from extract_feature import Feature
import numpy as np
import os
import pandas as pd
import pymysql


def check_sentences(results):
    new_results = []
    for r in results:
        if r[4] == '' or r[4] == ' ':  # 空内容，删掉
            continue
        temp = list(r)  # 转list，以便将过长的句子截短
        if len(temp[4]) > 3000:
            temp[4] = temp[4][:3000]
        new_results.append(temp)
    return new_results


def get_train_vector(batch_size=16):
    # if vector file can not found:
    # load raw train data from database, transform it to vector/features, and store to local disk (DataFrame).
    print("Start to transform train_data to vector.")
    transformer = Feature()
    train_vector = pd.DataFrame()
    db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
    cursor = db.cursor()
    total_num = cursor.execute("select * from `train_table`")
    count = 0
    while True:
        batch = cursor.fetchmany(batch_size)
        if len(batch) == 0:
            break
        df = pd.DataFrame(check_sentences(batch), columns=['pid', 'is_init', 'is_reply', 'img_num', 'content',
                                                           'reply_content', 'SIS', 'PIS', 'SES', 'PES', 'COM'])
        sentences = df.iloc[:, 4]
        feature = transformer.joint_feature(list(sentences))
        init_reply_img = df.iloc[:, 1:4]
        tags = df.iloc[:, -5:]
        vector = pd.concat([feature, init_reply_img, tags], axis=1)
        train_vector = train_vector.append(vector)
        count += len(batch)
        print('\r[%s%s] %s / %s' %
              (('#' * int(50 * count / total_num)), ' ' * (50 - int(50 * count / total_num)), count, total_num),
              end='')
    train_vector.to_pickle('./data/train_vector.df')
    db.close()
    print("\nTransform Vector Complete.")
    return train_vector


def run(force_trans=0):
    if os.path.exists('./data/train_vector.df') and not force_trans:
        df = pd.read_pickle('./data/train_vector.df')
    else:
        df = get_train_vector()
    feature = df.iloc[:, :-5]
    tags = df.iloc[:, -5:]


if __name__ == '__main__':
    run(1)
