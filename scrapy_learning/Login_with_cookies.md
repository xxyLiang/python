>## 1.直接携带cookies请求页面
首先，start_urls默认通过scrapy.Spider中的start_requests()函数进行请求，可通过重写start_requests()来使第一次请求携带cookies信息。

```python
class mySpider(scrapy.Spider):
	name = 'baidu'
	allowed_domains = ['baidu.com']
	start_urls = ['http://www.baidu.com']

	def start_requests(self):
		cookies = 'anonymid=sjsjsjs; depovince=GW; jebecookies=wowowj'
		# 传递的cookies需是字典形式
		cookies ={i.split('=')[0]:i.split('=')[1] for i in cookies.split(';')}
		for url in self.start.urls:
			yield scrapy.Request(url, callback=self.parse, cookies=cookies)

	def parse(self, response):
		pass

```

>## 2.找到发送post请求的url地址，带上信息，发送请求
表单内容格式需从网络调试工具中找，如下面的例子从session中找到表单格式

```python
class GithubSpider(scrapy.Spider):
	name = "github"
	allowed_domains = ['github.com']
	start_urls = ['https://github.com/login']

	def parse(self, response):
		authenticity_token = response.xpath("//input[@name=’authenticity_token’]/@value").extract_first()
		utf8 = response.xpath("//input[@name=’utf8’]/@value").extract_first()
		commit = response.xpath("//input[@name=’commit’]/@value").extract_first()
		post_data = dict(
			login='abc',
			password='123',
			authenticity_token=authenticity_token
			utf8=utf8,
			commit=commit
		)
		yield scrapy.FormRequest(
			'https://github.com/session', 
			formdata=post_data, 
			callback=self.after_login
		)

	def after_login(self, response):
		pass

```

### 若from表单中有action，则可自动寻找url地址，只需要传递用户名和密码的表单，如  
```html
<form accept-charset="UTF-8" action="/session" method="post">
```

```python
class Github2Spider(scrapy.Spider):
	name = "github2"
	allowed_domains = ['github.com']
	start_urls = ['https://github.com/login']
	
	def parse(self, response):
		yield scrapy.FormRequest.from_response(
			response, #自动地从response中寻找from表单
			formdata={"login":"abc","password":"123"},
			callback=self.after_login
		)	#还有其他参数，可用于在网页拥有多表单时，定位特定表单
		
	def after_login(self, response):
		pass
```
