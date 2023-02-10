import matplotlib.pyplot as plt
import pandas as pd
import pymysql
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import math

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()

cursor.execute("select uid, SIS_percent, PIS_percent, SES_percent, PES_percent, COM_percent from users_30")
rs = cursor.fetchall()
df = pd.DataFrame(rs, columns=['uid', 'SIS', 'PIS', 'SES', 'PES', 'COM'])

s = df.iloc[:, 1:].sum(axis=1)
s[s == 0] = 1
df['SIS'] /= s
df['PIS'] /= s
df['SES'] /= s
df['PES'] /= s
df['COM'] /= s

km = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=100)
label = km.fit_predict(df.iloc[:, 1:])
df = pd.concat([df, pd.DataFrame(label, columns=['class5_1'])], axis=1)
df.to_excel(r'C:\Users\65113\Desktop\cluster5.xlsx')

# distortions = []
# dbi = {}
# for i in range(1, 11):
#     km = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=100)
#     km.fit(df.iloc[:, 1:])
#     distortions.append(km.inertia_)
# plt.plot(range(1, 11), distortions, marker='o')
# plt.xlabel("k")
# plt.ylabel("SSE")
# plt.show()
#     label = km.fit_predict(df.iloc[:, 1:])
#     dbi[i] = davies_bouldin_score(df.iloc[:, 1:], label)
# print(dbi)


