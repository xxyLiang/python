# -*- coding: utf-8 -*-
# 分布式爬虫爬取当当图书
import scrapy
from scrapy_redis.spiders import RedisSpider
from copy import deepcopy
import urllib

class DangdangSpider(RedisSpider):  #继承RedisSpider类
    name = 'dangdang'
    allowed_domains = ['book.dangdang.com', 'category.dangdang.com']
    # start_urls = ['http://book.dangdang.com/']
    redis_key = "dangdang"
    # 可以看作是一个队列，redis_key是队列标识
    # 在redis_cli中输入`lpush dangdang http://book.dangdang.com/`
    # 那么start_url被push入队列，第一台电脑pop操作获得url，有且仅有一台电脑获得url

    def parse(self, response):
        # 大分类分组
        div_list = response.xpath("//div[@class='con flq_body']/div")
        for div in div_list:
            item = {}
            item['b_cate'] = div.xpath("./dl/dt//text()").extract()
            item['b_cate'] = [i.strip() for i in item['b_cate'] if len(i.strip()) > 0]
            # 有多个字符串，去掉字符串的空白和空字符串
            # 中间分类分组
            dl_list = div.xpath("./div//dl[@class='inner_dl']|dl[@class='inner_dl last']")
            for dl in dl_list:
                item['m_cate'] = dl.xpath("./dt//text()").extract()
                item['m_cate'] = [i.strip() for i in item['m_cate'] if len(i.strip()) > 0]
                # 小分类分组
                a_list = dl.xpath("./dd/a")
                for a in a_list:
                    item['s_href'] = a.xpath("./@href").extract_first()
                    item['s_cate'] = a.xpath("./text()").extract_first()
                    if item['s_href'] is not None:
                         yield scrapy.Request(
                            item['s_href'],
                            callback=self.parse_book_list,
                            meta={'item': deepcopy(item)}
                        )

    def parse_book_list(self, response):
        item = response.meta['item']
        li_list = response.xpath("//ul[@class='bigimg']/li")
        for li in li_list:
            item['book_name'] = li.xpath("./p[@class='name']/a/@title").extract_first()
            item['book_author'] = li.xpath("./p[@class='search_book_author']/span[1]/a/text()").extract()
            # xpath中的span[1]就是第一个span，而非第二个
            item['book_price'] = li.xpath(".//span[@class='search_now_price']/text()").extract_first()
            print(item)
        # 下一页
        next_url = response.xpath("//li[@class='next']/a/@href").extract_first()
        if next_url is not None:
            next_url = urllib.parse.urljoin(response.url, next_url)
            yield scrapy.Request(
                next_url,
                callback=self.parse_book_list,
                meta={'item': item}
            )
