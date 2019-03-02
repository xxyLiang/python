## 数据提取方法

>### json
- 数据交换格式，看起来像python类型（列表，字典）的字符串
- 哪里会返回json的数据？
  - 浏览器切换到手机版
  - 抓包app
- 使用json之前需要导入
- json.loads
  - 把json字符串转化为pythonl欸行
  - `json.loads(json字符串)`
- json.dumps
  - 把python类型转化为json字符串
  - json.dumps({})
  - json.dumps(ret1, ensure_ascii=False, indent=4)
    - ensure_ascii：让中文显示成中文
    - indent：能够让下一行在上一行的基础上空格