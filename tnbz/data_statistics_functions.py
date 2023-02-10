import math

import pymysql
import traceback
from datetime import timedelta
import pandas as pd
import numpy as np

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()

LOSS_RATIO = 2.25
GAIN_POW = 0.88
LOSS_POW = 0.88


def progress(present, total):
    block = 100
    print('\r[%s%s]  %s / %s' %
          (('#' * int(block * present / total)), ' ' * (block - int(block * present / total)), present, total),
          end='')


def set_crawl_threads():
    sql = "update `threads` set will_crawl = 1 where author_id=%s and (TO_DAYS(publish_date) - TO_DAYS(%s) <= 30)"

    cursor.execute("select author_id, publish_date from threads where new_user=1")
    ts = cursor.fetchall()
    for t in ts:
        try:
            cursor.execute(sql, (t[0], t[1]))
            db.commit()
        except:
            traceback.print_exc()
            db.rollback()


def fill_thread_statistics_support_type():

    def get_primary_second_support(s):
        sis = s[0]
        pis = s[1]
        ses = s[2]
        pes = s[3]
        com = s[4]

        s1 = None
        s2 = None

        # 优先级：SIS > SES > PES > PIS > COM
        if sis == 1:
            s1 = 'SIS'
        if ses == 1:
            if s1 is None:
                s1 = 'SES'
            else:
                s2 = 'SES'
        if pes == 1:
            if s1 is None:
                s1 = 'PES'
            elif s2 is None:
                s2 = 'PES'
            else:
                return s1, s2  # 已有两种类型了，后面的没必要验证了
        if pis == 1:
            if s1 is None:
                s1 = 'PIS'
            elif s2 is None:
                s2 = 'PIS'
            else:
                return s1, s2
        if com == 1:
            if s1 is None:
                s1 = 'COM'
            elif s2 is None:
                s2 = 'COM'
        return s1, s2

    sql_query = "select tid from thread_statistics"
    cursor.execute(sql_query)
    rs = cursor.fetchall()

    sql_query2 = "select SIS, PIS, SES, PES, COM from posts where tid=%s and is_initiate_post=1"
    sql_update = "update thread_statistics set primary_support_type=%s, second_support_type=%s where tid=%s"
    for r in rs:
        a = cursor.execute(sql_query2, r[0])
        if a != 1:
            continue
        supports = cursor.fetchall()[0]

        primary, second = get_primary_second_support(supports)

        try:
            cursor.execute(sql_update, (primary, second, r[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()


def match_type(t):
    if t == 'SIS':
        match = 'PIS'
    elif t == 'SES':
        match = 'PES'
    elif t == 'COM':
        match = 'COM'
    elif t == 'PIS':
        match = 'SIS'
    else:
        match = 'SES'
    return match


def fill_thread_statistics_support_match_num():
    total = cursor.execute("select tid, users.last_active_time, primary_support_type, second_support_type "
                           "from thread_statistics t left join users on t.uid=users.uid")
    rs = cursor.fetchall()

    sql_update = "update thread_statistics set support_match_total=%s, support_match_1=%s, support_match_2=%s " \
                 "where tid=%s"
    count = 0
    for r in rs:
        count += 1
        total_count = match1_count = match2_count = None
        if r[2] is not None:
            if r[1] is None:
                continue
            fix_time = r[1] + timedelta(hours=2)
            cursor.execute("select pid, SIS, PIS, SES, PES, COM from posts where tid=%s and is_thread_publisher=0 "
                           "and publish_time<%s", (r[0], str(fix_time)))
            ts = cursor.fetchall()
            df = pd.DataFrame(ts, columns=['pid', 'SIS', 'PIS', 'SES', 'PES', 'COM'])
            match1 = match_type(r[2])
            match1_count = len(df[df[match1] == 1])
            # match1_count = len(df[df[match1] == 1]) if match1 is not None else len(df[df.iloc[:, 1:].any(axis=1)])
            total_count = match1_count
            if r[3] is not None:
                match2 = match_type(r[3])
                match2_count = len(df[df[match2] == 1])
                total_count = len(df[df[[match1, match2]].any(axis=1)])
        try:
            cursor.execute(sql_update, (total_count, match1_count, match2_count, r[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
        progress(count, total)


def fill_thread_statistics_wordcount():
    total = cursor.execute("select tid, users.last_active_time from thread_statistics t left join users on "
                           "t.uid=users.uid")
    threads = cursor.fetchall()

    count = 0
    for thread in threads:
        count += 1
        fix_time = thread[1] + timedelta(hours=2)
        try:
            cursor.execute(
                "select is_initiate_post, is_thread_publisher, content from posts where tid=%s and publish_time<%s",
                (thread[0], str(fix_time)))
            ps = cursor.fetchall()
        except:
            traceback.print_exc()
            continue
        df = pd.DataFrame(ps, columns=['init', 'publisher', 'content'])
        publisher_posts = df['publisher'].sum()
        received_posts = len(df) - publisher_posts
        init_words = df[df['init'] == 1]['content'].sum()
        r_words = df[df['publisher'] == 0]['content'].sum()
        init_post_words = len(init_words) if type(init_words) is str else 0
        received_words = len(r_words) if type(r_words) is str else 0
        try:
            cursor.execute("update thread_statistics set publisher_posts=%s, init_post_words=%s, received_posts=%s, "
                           "received_words=%s where tid=%s",
                           (publisher_posts, init_post_words, received_posts, received_words, thread[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
            return -1
        progress(count, total)


def fill_users_support_type_percentage():
    total = cursor.execute("select uid, first_thread_time from users_30")
    us = cursor.fetchall()

    count = 0
    for u in us:
        count += 1
        cursor.execute("select SIS, PIS, SES, PES, COM from posts where author_id=%s and is_initiate_post=1", u[0])
        ts = cursor.fetchall()
        df = pd.DataFrame(ts, columns=['SIS', 'PIS', 'SES', 'PES', 'COM'])
        n = len(df)
        sis = df['SIS'].sum() / n
        pis = df['PIS'].sum() / n
        ses = df['SES'].sum() / n
        pes = df['PES'].sum() / n
        com = df['COM'].sum() / n

        biggest = max((sis, pis, ses, pes, com))
        if biggest == 0:
            main_type = 0
        else:
            if ses == biggest:
                main_type = 3
            elif pes == biggest:
                main_type = 4
            elif sis == biggest:
                main_type = 1
            elif pis == biggest:
                main_type = 2
            else:
                main_type = 5
        try:
            cursor.execute(
                "update users_30 set threads_num=%s, SIS_percent=%s, PIS_percent=%s, SES_percent=%s, PES_percent=%s, "
                "COM_percent=%s, main_support_type=%s where uid=%s", (n, sis, pis, ses, pes, com, main_type, u[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
            return -1
        progress(count, total)


def fill_users_received_support_num():
    total = cursor.execute("select uid from users_30")
    us = cursor.fetchall()

    count = 0
    for u in us:
        cursor.execute("select is_thread_publisher, SIS, PIS, SES, PES, COM from posts where tid in "
                       "(select tid from threads where author_id=%s)", u[0])
        rs = cursor.fetchall()
        df = pd.DataFrame(rs, columns=['self', 'SIS', 'PIS', 'SES', 'PES', 'COM'])
        others = df[df['self'] == 0]
        n = len(others)
        if n == 0:
            sis = pis = ses = pes = com = 0
        else:
            sis = others['SIS'].sum()
            pis = others['PIS'].sum()
            ses = others['SES'].sum()
            pes = others['PES'].sum()
            com = others['COM'].sum()
        self_post = len(df) - n
        try:
            cursor.execute(
                "update users_30 set received_post_num=%s, self_post_num=%s, received_SIS=%s, received_PIS=%s, "
                "received_SES=%s, received_PES=%s, received_COM=%s where uid=%s",
                (n, self_post, sis, pis, ses, pes, com, u[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
            return -1
        count += 1
        progress(count, total)


def fill_users_cost():
    total = cursor.execute("select uid, first_thread_time, first_thread_id, last_active_time from users_cost")
    us = cursor.fetchall()

    count = 0
    for u in us:
        cursor.execute("select tid, is_initiate_post, is_thread_publisher, img_num, publish_time "
                       "from posts where tid in "
                       "(select tid from threads where author_id=%s and DATEDIFF(publish_date, %s)<=7)",
                       (u[0], str(u[1])))
        rs = cursor.fetchall()
        df = pd.DataFrame(rs, columns=['tid', 'init', 'self', 'img', 'time'])
        df = pd.concat([df, pd.DataFrame(columns=['wait'])], axis=1)
        threads = df.tid.value_counts().keys()

        wait_column = pd.Series(dtype=object)

        for thread in threads:
            publish_time = min(df[df.tid == thread]['time'])
            wait = df[df.tid == thread]['time'] - publish_time
            wait_column = pd.concat([wait_column, wait])
        df.wait = wait_column

        thread_has_reply = 0
        thread_first_reply_wait = 0
        for thread in threads:  # 第一次遍历难以带上self来判断是否是其他人回复
            replies = df[(df['tid'] == thread) & (df['self'] == 0)]
            if not replies.empty:  # which means the thread has reply
                thread_has_reply += 1
                thread_first_reply_wait += min(replies['wait']).total_seconds() / 3600
        avg_first_reply_wait_hour = thread_first_reply_wait / thread_has_reply if thread_has_reply > 0 else 99

        first_thread_reply = df[(df.tid == u[2]) & (df.self == 0)]
        very_first_reply_wait_hour = min(
            first_thread_reply['wait']).total_seconds() / 3600 if not first_thread_reply.empty else 99

        first_day_reply = len(df[(df['self'] == 0) & (df['wait'] < timedelta(days=1))])
        three_day_reply = len(df[(df['self'] == 0) & (df['wait'] < timedelta(days=3))]) - first_day_reply

        img_num = df[df['self'] == 1]['img'].sum()
        img_num = math.log2(img_num + 1)

        try:
            cursor.execute("update users_cost set user_first_reply_wait_hour=%s, avg_first_reply_wait_hour=%s, "
                           "first_day_reply=%s, three_days_reply_increase=%s, log_self_img=%s where uid=%s",
                           (very_first_reply_wait_hour, avg_first_reply_wait_hour, first_day_reply, three_day_reply,
                            img_num, u[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
            return -1

        count += 1
        progress(count, total)


def fill_users_cost_self_word_count():
    total = cursor.execute("select uid, first_thread_time from users_cost")
    us = cursor.fetchall()

    count = 0
    for u in us:
        cursor.execute("select content from posts where tid in "
                       "(select tid from threads where author_id=%s and DATEDIFF(publish_date, %s)<=7) "
                       "and is_thread_publisher=1",
                       (u[0], str(u[1])))
        rs = cursor.fetchall()
        words = 0
        for r in rs:
            words += len(r[0])
        words = math.log10(words + 1)

        try:
            cursor.execute("update users_cost set lg_self_post_word=%s where uid=%s",
                           (words, u[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
            return -1

        count += 1
        progress(count, total)


def fill_thread_statistics_score(match_ratio=2):
    total = cursor.execute("select tid, primary_support_type, second_support_type, uid "
                           "from thread_statistics order by publish_time asc")
    ts = cursor.fetchall()

    hour6 = timedelta(hours=6)
    hour24 = timedelta(hours=24)
    hour48 = timedelta(hours=48)

    user_dict = {}          # 用于统计帖子是用户发的第几个帖子
    count = 0
    for t in ts:
        if t[3] in user_dict:
            user_dict[t[3]] += 1
        else:
            user_dict[t[3]] = 1
        thread_rank = user_dict[t[3]]

        cursor.execute(
            "select is_initiate_post, is_thread_publisher, img_num, content, publish_time, SIS, PIS, SES, PES, COM "
            "from posts where tid=%s", t[0])
        p = cursor.fetchall()
        df = pd.DataFrame(p, columns=['init', 'self', 'img', 'content', 'time', 'SIS', 'PIS', 'SES', 'PES', 'COM'])
        df = pd.concat([df, pd.DataFrame(columns=['wait', 'timeScore', 'score'])], axis=1)

        publish_time = df.iloc[0]['time']
        df.wait = df['time'] - publish_time

        df.loc[df.wait <= hour6, 'timeScore'] = 1
        df.loc[(df.wait > hour6) & (df.wait <= hour24), 'timeScore'] = 0.5
        df.loc[(df.wait > hour24) & (df.wait <= hour48), 'timeScore'] = 0.2
        df.loc[df.timeScore.isnull(), 'timeScore'] = 0.1

        df.score = (df.content.apply(lambda x: math.log10(len(x)+1)) + 2 * df.img.apply(lambda x: math.log2(x+1))) * df.timeScore

        losses = df.loc[df.init == 1, 'score'].sum() * (1 + 1/thread_rank)          # 发帖成本随发帖次数下降
        if t[1] is None:  # primary_support_type
            gains = df.loc[df.self == 0, 'score'].sum()
        else:
            match1 = match_type(t[1])
            if t[2] is None:
                match_post = df[match1] == 1
            else:
                match2 = match_type(t[2])
                match_post = df[[match1, match2]].any(axis=1)
            gains = df.loc[(df.self == 0) & match_post, 'score'].sum() * match_ratio \
                + df.loc[(df.self == 0) & ~match_post, 'score'].sum()

        score = cal_score(gains, losses)

        try:
            cursor.execute("update thread_statistics set gains=%s, losses=%s, score=%s where tid=%s",
                           (gains, losses, score, t[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
            return -1

        count += 1
        progress(count, total)


def cal_score(gains, losses):
    return gains ** GAIN_POW - LOSS_RATIO * losses ** LOSS_POW


def cal_user_support_score(threads):
    if len(threads) == 0:
        # return 0
        return {'gains': 0, 'losses': 0}

    day2 = timedelta(days=2)
    last_time = None
    last_bracket = {'gains': 0, 'losses': 0}
    brackets = 0
    # total_score = 0
    total_score = {'gains': 0, 'losses': 0}
    for t in threads.iterrows():        # t[0] -- row id; t[1] -- Series
        if last_time is not None and (t[1]['time'] - last_time) > day2:
            # total_score += cal_score(last_bracket['gains'], last_bracket['losses'])
            total_score['gains'] += last_bracket['gains'] ** GAIN_POW
            total_score['losses'] += last_bracket['losses'] ** LOSS_POW
            last_bracket = {'gains': 0, 'losses': 0}
            brackets += 1
        last_bracket['gains'] += t[1]['gains']
        last_bracket['losses'] += t[1]['losses']
        last_time = t[1]['time']
    # total_score += cal_score(last_bracket['gains'], last_bracket['losses'])
    total_score['gains'] += last_bracket['gains'] ** GAIN_POW
    total_score['losses'] += last_bracket['losses'] ** LOSS_POW
    brackets += 1
    return total_score


def fill_users_category_score_with_ma():
    total = cursor.execute("select uid, first_thread_time from users_30")
    us = cursor.fetchall()

    count = 0
    for u in us:

        cursor.execute("select tid, publish_time, primary_support_type, second_support_type, gains, losses "
                       "from thread_statistics where uid=%s and DATEDIFF(publish_time, %s)<=30", (u[0], str(u[1])))
        ts = cursor.fetchall()
        df = pd.DataFrame(ts, columns=['tid', 'time', 's1', 's2', 'gains', 'losses'])

        # is_thread = df[(df.s1 == 'SIS') | (df.s1 == 'PIS') | (df.s2 == 'SIS') | (df.s2 == 'PIS')]
        # es_thread = df[(df.s1 == 'SES') | (df.s1 == 'PES') | (df.s2 == 'SES') | (df.s2 == 'PES')]
        is_thread = df[(df.s1 == 'SIS') | (df.s2 == 'SIS')]
        es_thread = df[(df.s1 == 'SES') | (df.s2 == 'SES')]
        com_thread = df[(df.s1 == 'COM') | (df.s1 == 'COM')]

        is_score = cal_user_support_score(is_thread)
        es_score = cal_user_support_score(es_thread)
        com_score = cal_user_support_score(com_thread)

        try:
            # cursor.execute("update users_30 set IS_score=%s, ES_score=%s, COM_score=%s where uid=%s",
            #                (is_score, es_score, com_score, u[0]))
            cursor.execute("update users_30_copy1 set IS_gain=%s, IS_loss=%s, ES_gain=%s, ES_loss=%s, "
                           "COM_gain=%s, COM_loss=%s where uid=%s",
                           (is_score['gains'], is_score['losses'], es_score['gains'], es_score['losses'],
                            com_score['gains'], com_score['losses'], u[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
            return -1

        count += 1
        progress(count, total)


def users_diff_support_statistics():

    def func(posts, uid, t):
        sis = posts['SIS'].sum()
        pis = posts['PIS'].sum()
        ses = posts['SES'].sum()
        pes = posts['PES'].sum()
        com = posts['COM'].sum()

        try:
            cursor.execute("insert into `users_%s`(uid, rSIS, rPIS, rSES, rPES, rCOM) values('%s', %d, %d, %d, %d, %d)" % (
                t, uid, sis, pis, ses, pes, com))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()

    total = cursor.execute("select uid from users")
    us = cursor.fetchall()

    count = 0
    for u in us:
        cursor.execute(
            "select t.primary_support_type, t.second_support_type, p.SIS, p.PIS, p.SES, p.PES, p.COM from "
            "posts p left join thread_statistics t on p.tid=t.tid "
            "where t.uid=%s and is_thread_publisher=0", u[0])
        rs = cursor.fetchall()
        df = pd.DataFrame(rs, columns=['type1', 'type2', 'SIS', 'PIS', 'SES', 'PES', 'COM'])

        # SIS
        sis_post = df[(df.type1 == 'SIS') | (df.type2 == 'SIS')]
        ses_post = df[(df.type1 == 'SES') | (df.type2 == 'SES')]
        com_post = df[(df.type1 == 'COM') | (df.type2 == 'COM')]

        if len(sis_post) > 0:
            func(sis_post, u[0], 'sis')
        if len(ses_post) > 0:
            func(ses_post, u[0], 'ses')
        if len(com_post) > 0:
            func(com_post, u[0], 'com')

        count += 1
        progress(count, total)


def diff_support_thread_received_posts_statistics():
    total = cursor.execute("select tid, publisher_posts, init_post_words, received_posts, received_words, "
                           "primary_support_type, second_support_type from thread_statistics")
    ts = cursor.fetchall()

    d = {
        'SIS': {'count': 0, 'self_posts': 0, 'self_words': 0, 'posts': 0, 'r_words': 0},
        'PIS': {'count': 0, 'self_posts': 0, 'self_words': 0, 'posts': 0, 'r_words': 0},
        'SES': {'count': 0, 'self_posts': 0, 'self_words': 0, 'posts': 0, 'r_words': 0},
        'PES': {'count': 0, 'self_posts': 0, 'self_words': 0, 'posts': 0, 'r_words': 0},
        'COM': {'count': 0, 'self_posts': 0, 'self_words': 0, 'posts': 0, 'r_words': 0},
    }

    df = pd.DataFrame(ts, columns=['tid', 'sp', 'sw', 'rp', 'rw', 'type1', 'type2'])
    for index, row in df.iterrows():
        if row['type1'] is not None:
            d[row['type1']]['count'] += 1
            d[row['type1']]['self_posts'] += row['sp']
            d[row['type1']]['self_words'] += row['sw']
            d[row['type1']]['posts'] += row['rp']
            d[row['type1']]['r_words'] += row['rw']
            if row['type2'] is not None:
                d[row['type2']]['count'] += 1
                d[row['type2']]['self_posts'] += row['sp']
                d[row['type2']]['self_words'] += row['sw']
                d[row['type2']]['posts'] += row['rp']
                d[row['type2']]['r_words'] += row['rw']
    for i in d:
        print(i)
        print(d[i])


if __name__ == '__main__':
    diff_support_thread_received_posts_statistics()
    ...
