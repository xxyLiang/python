## 数据提取方法

>### json
- 数据交换格式，看起来像python类型（列表，字典）的字符串
- 哪里会返回json的数据？
  - 浏览器切换到手机版
  - 抓包app
- 使用json之前需要导入
- json.loads
  - 把json字符串转化为python类型
  - `json.loads(json字符串)`
- json.dumps
  - 把python类型转化为json字符串
  - json.dumps({})
  - json.dumps(ret1, ensure_ascii=False, indent=4)
    - ensure_ascii：让中文显示成中文
    - indent：能够让下一行在上一行的基础上空格

### 豆瓣电视剧.py
```python
import requests
import json
from retrying import retry

class DoubanSpider:

    def __init__(self):
        self.temp_url = "https://m.douban.com/rexxar/api/v2/subject_collection/tv_american/items?os=ios&for_mobile=1&start={}&count=18&loc_id=108288&_=0"

    @retry(stop_max_attempt_number=3)
    def parse_url(self, url):
        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 11_0 like Mac OS X) AppleWebKit/604.1.38 (KHTML, like Gecko) Version/11.0 Mobile/15A372 Safari/604.1",
            "Referer": "https://m.douban.com/tv/american"}
        response = requests.get(url, headers=headers, timeout=5)
        return response.content.decode()

    def get_content_list(self, html_str):  # 提取数据
        dict_data = json.loads(html_str)
        # !!!注意response的文本是json还是jsonp字符串，url的参数字段可能含有'&callback=jsonp'，返回的是jsonp字符串!!!
        content_list = dict_data["subject_collection_items"]
        total = dict_data["total"]
        return content_list, total

    def save_content_list(self, content_list):
        with open("douban.json", "a", encoding="utf-8") as f:
            for content in content_list:
                f.write(json.dumps(content, ensure_ascii=False))
                f.write("\n")
            print("保存成功")

    def run(self):
        num = 0
        total = 100
        while num < total:
            start_url = self.temp_url.format(num)
            html_str = self.parse_url(start_url)
            content_list, total = self.get_content_list(html_str)
            self.save_content_list(content_list)
            num += 18


if __name__ == "__main__":
    douban = DoubanSpider()
    douban.run()
```

<br/>

>### xpath和lxml
- xpath
  - 一门从html中提取数据的语言
- xpath语法
  - xpath helper插件：帮助我们从`elements`中定位数据
  - `/html/head/meta`：选择节点（标签），能够选中html下的head下的所有的meta标签
  - `//标签`：能够从任意节点开始选择
  - `@属性=属性名`
  - `text()`获取文本
- lxml
  - 安装：pip install lxml
  - 使用
    ```python
    from lxml import etree
    element = etree.HTML("html字符串")
    element.xpath("xpath格式")  #返回一个list，string或Element格式
    ```