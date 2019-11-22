# -*- coding: utf-8 -*-
import scrapy


class DingxiangSpider(scrapy.Spider):
    name = 'dingxiang'
    allowed_domains = ['ask.dxy.com']
    start_urls = ['http://ask.dxy.com/']

    def parse(self, response):
        pass
