import requests
from lxml import etree
import re
import random
import string
import traceback
from fake_useragent import UserAgent

ua = UserAgent()
base = 'https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=4&orderby=dateline&orderby=dateline&filter=author&page=%d'


def parse():
    for page in range(2, 4):   # 2016年1520页左右
        referer = base % (page-1)
        page_url = base % page
        headers = {'Host': 'bbs.tnbz.com',
                   'Referer': referer,
                   'DNT': '1',
                   'User-Agent': ua.random}
        try:
            response = requests.get(page_url, headers=headers)
            response.raise_for_status()
            response.encoding = 'utf-8'
            tree = etree.HTML(response.content)
            threads = tree.xpath('//tbody[starts-with(@id, "normalthread_")]')

            for thread in threads:
                try:
                    thread_url = thread.xpath('.//td[@class="icn"]/a/@href')[0]
                    r = re.search(r'(tid=)(\d+)', thread_url)
                    tid = random_id() if r is None else r.group(2)

                except Exception as e:
                    traceback.print_exc()
                    continue

        except Exception as e:
            traceback.print_exc()


def random_id():
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))

