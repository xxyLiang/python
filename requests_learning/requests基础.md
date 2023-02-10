
### 发送get,post请求，获取响应
- response = requests.get(url) #发送get请求，请求url对应地址的相应
- response = requests.post(url, data={请求体的字典}) #发送post请求

### response的方法
- response.text
  - 该方式往往会出现乱码，出现乱码使用response.encoding = 'utf-8'
- response.content.decode()
  - 把响应的二进制字节转换为str类型
- response.requests.url #发送请求的url地址
- response.url #response响应的url地址
- response.requests.headers #请求头
- response.headers #响应请求

### 获得网页源码的正确打开方式
通过下面三种方式一定能够获得网页的正确解码之后的字符串
- response.content.decode()
- response.content.decode('gbk')
- response.text

### 发送headers请求
- 为了模拟浏览器，获得和浏览器一模一样的内容
```python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"
    "Referer": ""

response = requests.get(url, headers=headers)
}
```
<br/>

### 使用超时参数
- requests.get(url, headers=headers, timeout=3) #3秒内必须返回响应，否则会报错

### retrying模块
- pip install retrying
```python
from retrying import retry

# 装饰器
@retry(stop_max_attempt_number=3)
def fun1():
    print("this is func1")
    raise ValueError("this is test error)
```
<br/>

### 处理cookie相关的请求
- 直接携带cookie请求url地址
  - 1. cookie放在headers中
    ```python
    headers = {"User-Agent":"...", "Cookie":"Cookie字符串"}
    ```
  - 2. cookie字典传给cookies参数
    - requests.get(url, cookies=cookie_dict)

- 先发送post请求，获取cookie，带上cooked请求登录后的页面
  - 1. session = requests.session() #session具有的方法和requests一样
  - 2. session.post(url, data, headers) #服务器设置在本地的cookie会保存在session中
  - 3. session.get(url) #会带上之前保存在session中的cookie