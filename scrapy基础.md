
>### Scrapy命令行格式  

`scrapy <command> [option] [args]`



>### 常用命令

| 命令         | 说明               | 格式                                      |
| ------------ | ------------------ | ----------------------------------------- |
| startproject | 创建一个新工程     | scrapy startproject <name> [dir]          |
| genspider    | 创建一个爬虫       | scrapy genspider [option] <name> <domain> |
| settings     | 获得爬虫配置信息   | scrapy settings [option]                  |
| crawl        | 运行一个爬虫       | scrapy crawl <spider>                     |
| list         | 列出工程中所有爬虫 | scrapy list                               |


>### 基本模板
```python
import scrapy
class mySpider(scrapy.Spider):
	name = “…”
	allowed_domains = [‘baidu.com’]
	start_urls = [‘http://www.baidu.com’]

	def parse(self, response):
		tr_list = response.xpath(“…”)[1:-1]
		for tr in tr_list:
			item[‘1’] = tr.xpath(‘…’).extract_first()
			item[‘2’] = tr.xpath(‘…’).extract_first()
			……
			yield item

		next_url = response.xpath(‘…’).extract_first()
		if …:
			yield scrapy.Request(next_url, callback=self.parse)
```

>### scrapy.Request的知识点：

```python
scrapy.Request(url, [,callback, method=”GET”, headers, body, cookies, meta, dont_filter=False])
# 方括号表示可选参数，常用参数为：
# callback:指定传入的url交给哪个解析函数去处理
# meta:实现在不同解析函数中传递数据，meta会默认携带部分信息，比如下载延迟，请求深度等。如将item从parse传到parse1，parse使用meta={“item”:item}来传递，parse1使用response.meta[“item”]来获取）
# dont_filter:让scrapy的去重不会过滤当前url，scrapy默认有url去重的功能，对需要重复请求的url有重要用途。（如百度贴吧，页面会持续变化）

# 获取下一页：
……
	yield scrapy.Request(next_page_url, callback=self.parse)
……
```

>### request.response知识点

```python
response.body.decode()	#response的string，常用于：
re.findall("format"，response.body.decode())

item['name'] = response.xpath(…).extract()[0]
item['password'] = response.xpath(...).extract_first()
# extract()[0]用extract_first()来代替，返回一个string
```
