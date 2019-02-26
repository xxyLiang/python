## scrapy_redis模块用于防止重复爬取  
详情找百度，下面是一个例子
domz.project


# spider.py

```python
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class DmozSpider(CrawlSpider):
    """Follow categories and extract links."""
    name = 'dmoz'
    allowed_domains = ['dmoztools.net']
    start_urls = ['http://dmoztools.net/']
    
    rules = [
        Rule(LinkExtractor(
            restric_css('.top-cat', '.sub-cat', '.cat-item')
        ), callback='parse_directory', follow=True),
    ]
    
    def parse_directory(self, response):
        for div in response.css('.title-and-desc'):
            yield {
                'name':div.css('.site-title::text').extract_first()
                'description':div.css('.site-descr::text').extract_first()
                'link':div.css('a::attr(href)').extract_first()
            }
```

# settings.py

```python
SPIDER_MODULES = ['example.spiders']
NEWSPIDER_MODULE = 'example.spiders'

USER_AGENT = 'scrapy-redis (+https://github.com/rolando/scrapy-redis)'

DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"		#指定哪个去重办法给request对象去重
SCHEDULER = "scrapy_redis.scheduler.Scheduler"		#指定scheduler队列
SCHEDULER_PERSIST = True	#队列中的内容是否持久保存，为False的时候在关闭redis的时候清空
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderPriorityQueue"
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderQueue"
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderStack"

ITEM_PIPELINES = {
	'example.pipelines.ExamplePipeline':300,
	'scrapy_redis.pipelines.RedisPipeline':400,		#scrapy_redis实现的items保存到redis的pipeline
}

LOG_LEVEL = 'DEBUG'

DOWNLOAD_DELAY = 0.3

REDIS_URL = 'redis://127.0.0.1:6379'	# 指定redis的地址
# REDIS_HOST = "127.0.0.1"				# redis的地址可以写成如下形式
# REDIS_PORT = 6379
```