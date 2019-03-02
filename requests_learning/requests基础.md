
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

### 获得网页源码的正确打开方式（通过下面三种方式一定能够获得网页的正确解码之后的字符串）
- 1. response.content.decode()
- 2. response.content.decode('gbk)
- 3. response.text

### 发送headers请求
- 为了模拟浏览器，获得和浏览器一模一样的内容
```python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"
    "Referer": ""

response = requests.get(url, headers=headers)
}
```