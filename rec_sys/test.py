import time
import pandas as pd
import pymysql
import traceback
import numpy as np


db = pymysql.connect(host='localhost',
                     user='root',
                     password='651133439a',
                     database='rec_sys')
cursor = db.cursor()

cursor.execute("select pid, tid, `rank` from posts where tid in "
               "(select DISTINCT(tid) from posts where `rank` is NULL)"
               "order by tid, pid")
data = pd.DataFrame(cursor.fetchall(), columns=['pid', 'tid', 'rank'])

for tid, df in data.groupby('tid'):
    last_rank = 0
    for _, row in df.iterrows():
        if np.isnan(row['rank']):
            last_rank += 1
            try:
                cursor.execute("update posts set `rank`=%s where pid=%s", (last_rank, row['pid']))
                db.commit()
            except:
                db.rollback()
                traceback.print_exc()
        else:
            last_rank = row['rank']


import networkx as nx
nodes = [i for i in range(6)]
edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)]

G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)
c_degree = nx.degree_centrality(G)

