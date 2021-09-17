import re
import traceback
import time
import scrapy
import pymysql
from tmjy.items import TmjyItem
import requests
from tmjy.settings import ipPool


class PostsSpider(scrapy.Spider):
    name = 'posts'
    allowed_domains = ['bbs.tnbz.com']
    start_urls = ['https://bbs.tnbz.com/']
    db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
    cursor = db.cursor()
    # 运行前注意是否是正式表
    sql_get_threads_list = "select tid, forum, page, url from `threads` where will_crawl=1 and crawled=0"
    sql_update_crawled = "update `threads` set crawled=1, crawled_timestamp=%s where tid=%s"

    referer_base = 'https://bbs.tnbz.com/forum.php?mod=forumdisplay&fid=%d&orderby=dateline&orderby=dateline&filter=author&page=%d'
    fid = [0, 4, 2, 57, 9, 58]
    count = 0

    ips = requests.get('http://webapi.http.zhimacangku.com/getip?num=10&type=1&pro=0&city=0&yys=100026&port=11&pack=179910&ts=0&ys=0&cs=0&lb=1&sb=0&pb=45&mr=2&regions=')
    for ip in ips.text.strip('\r\n').split('\r\n'):
        ipPool.append('http://' + ip)

    def parse(self, response):
        self.cursor.execute(self.sql_get_threads_list)
        threads = self.cursor.fetchall()
        for thread in threads:
            referer = self.referer_base % (self.fid[thread[1]], thread[2])
            headers = {'Host': 'bbs.tnbz.com',
                       # "Accept-Encoding": "gzip,deflate",
                       'Referer': referer,
                       'DNT': '1'}
            yield scrapy.Request(
                thread[3],
                callback=self.parse_post_web,
                headers=headers,
                meta={'tid': thread[0], 'success_flag': 1}
            )

    def parse_post_web(self, response):
        tid = response.meta['tid']
        success_flag = response.meta['success_flag']
        posts = response.xpath('//table[starts-with(@id, "pid")]')
        if len(posts) == 0:
            success_flag = 0
            # try:
            #     ipPool.remove(response.request.meta['proxy'])
            # except:
            #     pass
            # time.sleep(15)
        print("tid:%s; Status:%d; Posts:%d;" % (tid, response.status, len(posts)))
        for post in posts:
            try:
                item = TmjyItem()
                item['tid'] = tid
                item['pid'] = re.search(r'\d+', post.attrib['id']).group()
                item['rank'] = post.xpath('.//div[@class="pi"]/strong/a/em/text()').extract_first()
                item['is_initiate_post'] = (item['rank'] == '1')
                item['is_thread_publisher'] = \
                    (item['rank'] == '1' or
                     len(post.xpath('.//img[starts-with(@id, "authicon")][contains(@src, "ico_lz")]')) > 0)
                item['img_num'] = len(post.xpath('.//img[contains(@id, "img")][@width]'))
                item['content'] = ' '.join(post.xpath('.//td[starts-with(@id, "postmessage_")]/text()').extract())
                item['author_id'] = re.search(r'\d+', post.xpath('.//div[@class="authi"]/a[@class="xw1"]/@href').extract_first()).group()
                item['author_nickname'] = post.xpath('.//div[@class="authi"]/a[@class="xw1"]/text()').extract_first()
                level = re.search(r'\[LV.(\d+|Master)]', ''.join(post.xpath('.//div[starts-with(@id, "favatar")]//p').extract()))
                item['author_level'] = level.group(1) if level is not None else None
                try:
                    item['reply_to_pid'] = re.search(r'(pid=)(\d+)', post.xpath('.//blockquote//a/@href').extract_first()).group(2)
                except:
                    item['reply_to_pid'] = None
                item['publish_time'] = re.search(
                    r'发表于(.+)',
                    post.xpath('.//em[starts-with(@id, "authorposton")]/text()').extract_first()
                ).group(1).strip()
                yield item

            except Exception as e:
                traceback.print_exc()
                success_flag = 0

        # 翻下一页
        next_page = response.xpath('//div[@class="pg"]/a[@class="nxt"]/@href').extract_first()
        if next_page is not None and re.match('http', next_page) is not None:
            yield scrapy.Request(
                next_page,
                callback=self.parse_post_web,
                headers={'Host': 'bbs.tnbz.com', 'Referer': response.url, 'DNT': '1'},
                meta={'tid': tid, 'success_flag': success_flag}
            )
        else:
            if success_flag == 1:
                try:
                    self.cursor.execute(
                        self.sql_update_crawled,
                        (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), tid)
                    )
                    self.db.commit()
                    self.count += 1
                    if self.count % 50 == 0:
                        print("%d threads' posts got." % self.count)
                except:
                    self.db.rollback()
                    traceback.print_exc()
