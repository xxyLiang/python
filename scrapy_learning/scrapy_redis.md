## scrapy_redis模块用于防止重复爬取  
>### redis的windows安装
1. 在cmd里输入  
   `pip3 install scrapy-redis`  
   `easy_install scrapy-redis`
2. 到网站下载redis客户端：[https://github.com/MicrosoftArchive/redis/releases]
3. cd到目录，运行redis：`redis-server redis.windows.conf`
4. 其他操作：`redis-cli`，具体百度

`key *` 列出所有变量
`TYPE variable` 返回变量类型
`ZCARD zset` 返回zset的成员个数
`SCARD set` 返回set的成员个数
`LLEN list` 返回list的成员个数
`ZRANGE zset 0 -1` 返回zset的所有成员
`SMEMBER set` 返回set的所有成员
`LRANGE list 0 -1` 返回list的所有成员
`DEL variable` 删除变量
`flushdb` 清空

 
下面是一个例子  

# spider.py

```python
import scrapy
from scrapy_redis.spiders import RedisSpider


class DangdangSpider(RedisSpider):  #继承RedisSpider类
    name = 'dangdang'
    allowed_domains = ['book.dangdang.com', 'category.dangdang.com']
    # start_urls = ['http://book.dangdang.com/']
    redis_key = "dangdang"
    # 可以看作是一个队列，redis_key是队列标识
    # 在redis_cli中输入`lpush dangdang http://book.dangdang.com/`
    # 那么start_url被push入队列，第一台电脑pop操作获得url，有且仅有一台电脑获得url

    def parse(self, response):
        pass

```

# settings.py

```python
SPIDER_MODULES = ['example.spiders']
NEWSPIDER_MODULE = 'example.spiders'

USER_AGENT = 'scrapy-redis (+https://github.com/rolando/scrapy-redis)'

# 以下四行是与普通scrapy的不同之处，换言之，只要启动了redis-server，在写好的scrapy程序中添加以下四行即可使用redis

DUPEFILTER_CLASS = "scrapy_redis.dupefilter.RFPDupeFilter"		#指定哪个去重办法给request对象去重
SCHEDULER = "scrapy_redis.scheduler.Scheduler"		#指定scheduler队列
SCHEDULER_PERSIST = True	#队列中的内容是否持久保存，为False的时候在关闭redis的时候清空
REDIS_URL = 'redis://127.0.0.1:6379'	# 指定redis的地址
# REDIS_HOST = "127.0.0.1"				# redis的地址可以写成如下形式
# REDIS_PORT = 6379
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderPriorityQueue"
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderQueue"
# SCHEDULER_QUEUE_CLASS = "scrapy_redis.queue.SpiderStack"

# 使用redis的pipeline时，需设置
ITEM_PIPELINES = {
	'example.pipelines.ExamplePipeline':300,
	'scrapy_redis.pipelines.RedisPipeline':400,		#scrapy_redis实现的items保存到redis的pipeline
}

LOG_LEVEL = 'DEBUG'

DOWNLOAD_DELAY = 0.3


```
