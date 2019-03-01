# 在spider中，settings能够通过self.settings的方式获取到

```python
class MySpider(scrapy.Spider):
	name = “myspider”
	start_urls = [‘…’]
	
	def parse(self, response):
		self.settings[“A_VARIABLE”]
		self.settings.get(“A_VARIABLE”)
```

# 在pipeline中获取settings的方法
```python
class MyscrapyPipeline(object):
	def open_spider(self, spider):
		spider.settings.get(“A_VARIABLE”)
```