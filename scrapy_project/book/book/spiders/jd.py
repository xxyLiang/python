# -*- coding: utf-8 -*-
import scrapy
from copy import deepcopy
from scrapy_redis.spiders import RedisSpider
import urllib
import re


class JdSpider(RedisSpider):
    name = 'jd'
    allowed_domains = ['list.jd.com']
    redis_key = "jd"
    # start_urls = ['https://book.jd.com/booksort.html']
    # 在redis_cli中输入`lpush dangdang https://book.jd.com/booksort.html`

    def parse(self, response):
        dt_list = response.xpath("//div[@class='mc']/dl/dt")    #大分类列表
        wanted_list = ["小说", "文学", "青春文学", "传记", "励志与成功",
                        "管理", "经济", "金融与投资", "历史", "心理学",
                        "社会科学", "科普读物", "计算机与互联网"]
        for dt in dt_list:
            item = {}
            item["b_cate"] = dt.xpath("./a/text()").extract_first()
            if item["b_cate"] not in wanted_list:
                continue
            em_list = dt.xpath("./following-sibling::dd[1]/em") #小分类列表
            for em in em_list:
                item["s_href"] = em.xpath("./a/@href").extract_first()
                item["s_href"] = re.findall(re.compile(r'com/(.+)\.html'), item["s_href"])[0]
                item["s_href"] = item["s_href"].replace('-', ',')
                item["s_cate"] = em.xpath("./a/text()").extract_first()
                if item["s_href"] is not None:
                    item["s_href"] = "https://list.jd.com/list.html?cat=" + item["s_href"] \
                                     + "&delivery=1&sort=sort_totalsales15_desc"
                    yield scrapy.Request(
                        item["s_href"],
                        callback=self.parse_book_list,
                        meta={"item": deepcopy(item)}
                    )

    def parse_book_list(self, response):
        item = response.meta["item"]
        li_list = response.xpath("//div[@id='plist']/ul/li")
        book_number_parse = 0
        for li in li_list:
            item["book_img"] = li.xpath(".//div[@class='p-img']//img/@src").extract_first()
            if item["book_img"] is None:
                item["book_img"] = li.xpath(".//div[@class='p-img']//img/@data-lazy-img").extract_first()
            item["book_img"] = "https:" + item["book_img"] if item["book_img"] is not None else None
            item["book_name"] = li.xpath(".//div[@class='p-name']/a/em/text()").extract_first().strip()
            # item["book_disc"] = li.xpath(".//div[@class='p-name']/a/i/text()").extract_first()
            # if item["book_disc"] is not None:
            #     item["book_disc"] = item["book_disc"].strip()
            item["book_author"] = li.xpath(".//span[@class='author_type_1']/a/text()").extract()
            item["book_author"] = ", ".join(item["book_author"])
            item["book_press"] = li.xpath(".//span[@class='p-bi-store']/a/@title").extract_first()
            item["book_id"] = li.xpath("./div/@data-sku").extract_first()
            book_number_parse += 1
            yield item
            if book_number_parse >= 50:
                return

        # 列表页翻页
        next_url = response.xpath("//a[@class='pn-next']/@href").extract_first()
        if next_url is not None:
            next_url = urllib.parse.urljoin(response.url, next_url)
            # 一种补全url的方式
            yield scrapy.Request(
                next_url,
                callback=self.parse_book_list,
                meta={"item": item}
            )


