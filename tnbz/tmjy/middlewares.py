# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter
import base64
from scrapy.downloadermiddlewares.httpproxy import HttpProxyMiddleware
from tmjy.settings import ipPool, count
import random
import requests
import time


# 代理服务器
proxyServer = "http://dyn.horocn.com:50000"

# 隧道身份信息
proxyUser = b"23AK1710860241468023"
proxyPass = b"sTo9QhVp31m9B1vH"
proxyAuth = "Basic " + base64.b64encode(proxyUser + b":" + proxyPass).decode()


class ProxyMiddleware(HttpProxyMiddleware):

    def process_request(self, request, spider):
        request.headers["Connection"] = "close"
        interval = time.time() - count['time']
        if len(ipPool) == 0 or interval > 70:
            ipPool.clear()
            ips = requests.get('http://webapi.http.zhimacangku.com/getip?num=10&type=1&pro=0&city=0&yys=100026&port=11&pack=179910&ts=0&ys=0&cs=0&lb=1&sb=0&pb=45&mr=2&regions=')
            for ip in ips.text.strip('\r\n').split('\r\n'):
                ipPool.append('http://' + ip)
            count['time'] = time.time()
        ip = random.choice(ipPool)
        request.meta['proxy'] = ip
        return None
        # request.meta["proxy"] = proxyServer
        # request.headers["Proxy-Authorization"] = proxyAuth
        # request.headers["Accept-Encoding"] = "gzip"

class TmjySpiderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the spider middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_spider_input(self, response, spider):
        # Called for each response that goes through the spider
        # middleware and into the spider.

        # Should return None or raise an exception.
        return None

    def process_spider_output(self, response, result, spider):
        # Called with the results returned from the Spider, after
        # it has processed the response.

        # Must return an iterable of Request, or item objects.
        for i in result:
            yield i

    def process_spider_exception(self, response, exception, spider):
        # Called when a spider or process_spider_input() method
        # (from other spider middleware) raises an exception.

        # Should return either None or an iterable of Request or item objects.
        pass

    def process_start_requests(self, start_requests, spider):
        # Called with the start requests of the spider, and works
        # similarly to the process_spider_output() method, except
        # that it doesn’t have a response associated.

        # Must return only requests (not items).
        for r in start_requests:
            yield r

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)


class TmjyDownloaderMiddleware:
    # Not all methods need to be defined. If a method is not defined,
    # scrapy acts as if the downloader middleware does not modify the
    # passed objects.

    @classmethod
    def from_crawler(cls, crawler):
        # This method is used by Scrapy to create your spiders.
        s = cls()
        crawler.signals.connect(s.spider_opened, signal=signals.spider_opened)
        return s

    def process_request(self, request, spider):
        # Called for each request that goes through the downloader
        # middleware.

        # Must either:
        # - return None: continue processing this request
        # - or return a Response object
        # - or return a Request object
        # - or raise IgnoreRequest: process_exception() methods of
        #   installed downloader middleware will be called
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # Called when a download handler or a process_request()
        # (from other downloader middleware) raises an exception.

        # Must either:
        # - return None: continue processing this exception
        # - return a Response object: stops process_exception() chain
        # - return a Request object: stops process_exception() chain
        pass

    def spider_opened(self, spider):
        spider.logger.info('Spider opened: %s' % spider.name)
