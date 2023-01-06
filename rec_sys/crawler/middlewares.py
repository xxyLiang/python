# Define here the models for your spider middleware
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/spider-middleware.html

from scrapy import signals
from crawler.settings import USER_AGENT
import random

# useful for handling different item types with a single interface
from itemadapter import is_item, ItemAdapter
import base64
from scrapy.downloadermiddlewares.httpproxy import HttpProxyMiddleware
import time


class FailLog:

    @classmethod
    def from_crawler(cls, crawler):
        return cls()

    def process_response(self, request, response, spider):
        if response.status >= 400:
            reason = response.status
            self._faillog(request, u'HTTPERROR', reason, spider)
        return response

    def process_exception(self, request, exception, spider):
        self._faillog(request, u'EXCEPTION', exception, spider)
        return request

    def _faillog(self, request, errorType, reason, spider):
        with open('logfail.log', 'a') as file:
            file.write("[%(error)s] %(url)s reason: %(reason)s headers:\n %(headers)s \r\n" % {
                'error': errorType,
                'url': request.url,
                'reason': reason,
                'headers': str(request.headers)})


class CrawlerDownloaderMiddleware:
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

        request.headers['User-Agent'] = random.choice(USER_AGENT)

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
