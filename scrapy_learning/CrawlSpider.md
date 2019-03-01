### 用于爬取网页上特定的链接，如详细内容页面、下一页的url  
### 生成crawlspider的命令：  
`scrapy genspider -t crawl [name] [url]`

```python
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class MyspiderSpider(CrawlSpider):	#继承自spiders的crawlspider类
	name = ‘…’
	allowed_domains = [‘…’]
	start_urls = [‘…’]
	#定义提取url地址规则
	rules = [Rule(LinkExtractor(
			allow='/web/site0/tab5420/info\d+\.htm'),
				callback=parse_item, 
	   			follow=True),
	]
	#LinkExtractor 链接提取器，提取url地址,allow地址不完整，会自动补充完整
	#callback 提取出来的url地址的response会交给callback处理
	#follow 当前url地址的响应是否重新进入rules来提取url地址

	#parse函数有特殊功能，不能定义
	def parse_item(self, response): 	
		pass
```
