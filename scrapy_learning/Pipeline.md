### 自定义Pipeline需要在settings中开启ITEM_PIPELINES并赋予距离（权重），越小优先级越高

>## 有三个基本函数

```python
class MyscrapyPipeline(object):

	def open_spider(self, spider):		#在爬虫开启的时候执行，仅执行一次
		self.file = open(spider.settings.get(“SAVE_FILE”, “./temp.json”, “w”)

	def close_spider(self, spider):		#在爬虫关闭的时候执行，仅执行一次
		self.file.close()

	def process_item(self, item, spider):
		……
		return item	
	#不return的情况下，另一个权重较低的pipeline就不会获取到该item
```