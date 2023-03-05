import os
import pymysql

prefix = os.path.expanduser("~") + '/Files/Recsys_data/'
filetype = '.pickle'

N_TOPICS = 20
HISTORY_THREAD_TAKEN_CNT = 10
VECTOR_DIM = 50

THREAD_CNT_LOW = 5
THREAD_CNT_HIGH = 10000

db = pymysql.connect(host='localhost',
                     user='root',
                     password='651133439a',
                     database='rec_sys')
cursor = db.cursor()
