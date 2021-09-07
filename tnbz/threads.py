import requests
from requests.adapters import HTTPAdapter
from lxml import etree
import re
import random
import string
import traceback
import pymysql
import time
from fake_useragent import UserAgent

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()
ua = UserAgent()
# 糖尿病2
# base = 'https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=4&orderby=dateline&orderby=dateline&filter=author&page=%d'
# 糖尿病1
# base = 'https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=2&orderby=dateline&orderby=dateline&filter=author&page=%d'
# 饮食与运动
# base = 'https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=57&orderby=dateline&orderby=dateline&filter=author&page=%d'
# 心情驿站
base = 'https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=9&orderby=dateline&orderby=dateline&filter=author&page=%d'

sql = 'insert into threads values(' + ('%s,'*14)[:-1] + ')'
a = [147]


def parse():
    total_count = 0
    fail = []
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=3))
    s.mount('https://', HTTPAdapter(max_retries=3))
    for page in a:
    # for page in range(1, 240):   # 2016年1520页左右
        time.sleep(2)
        count = 0
        referer = base % (page-1)
        page_url = base % page
        headers = {'Host': 'bbs.tnbz.com',
                   'Referer': referer,
                   'DNT': '1',
                   'User-Agent': ua.random}
        try:
            response = s.get(page_url, headers=headers, timeout=3)
            response.raise_for_status()
            response.encoding = 'utf-8'
            tree = etree.HTML(response.content)
            threads = tree.xpath('//tbody[starts-with(@id, "normalthread_")]')

            for thread in threads:
                try:
                    thread_url = thread.xpath('.//td[@class="icn"]/a/@href')[0]
                    r = re.search(r'(tid=)(\d+)', thread_url)
                    tid = r.group(2) if r is not None else random_id()      # tid 是主键，匹配不到就随机生成一个
                    title = thread.xpath('.//th/a[@class="s xst"]/text()')[0]
                    label_list = thread.xpath('.//th/em/a/text()')
                    label = label_list[0] if len(label_list) > 0 else ''
                    new_user = 1 if len(thread.xpath('.//th/img[@alt="新人帖"]')) > 0 else 0
                    author_id = re.search(r'(uid-)(\d+)', thread.xpath('.//td[@class="by"]/cite/a/@href')[0]).group(2)
                    author_nickname = thread.xpath('.//td[@class="by"]/cite/a/text()')[0]
                    publish_date, last_reply_time = thread.xpath('.//td[@class="by"]/em//text()')
                    replies = thread.xpath('.//td[@class="num"]/a/text()')[0]
                    reads = thread.xpath('.//td[@class="num"]/em/text()')[0]
                    item = (tid, 4, page, thread_url, title, label, new_user, author_id, author_nickname,
                            publish_date, replies, reads, last_reply_time, 0)
                except Exception as e:
                    traceback.print_exc()
                    continue
                try:
                    cursor.execute(sql, item)
                    db.commit()
                    count += 1
                except Exception as e:
                    db.rollback()

        except Exception as e:
            traceback.print_exc()
            fail.append(page)
        total_count += count
        print("Page %d finished, %d threads have been recorded, %d in total." % (page, count, total_count))
    print(fail)


def random_id():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))


if __name__ == '__main__':
    parse()

