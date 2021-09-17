import requests
import time

url='https://2021.ip138.com/'
proxyaddr = ""    #代理IP地址
proxyport = 57114               #代理IP端口
proxyusernm = "651133439"        #代理帐号
proxypasswd = "4201119!Dly"        #代理密码
#name = input();
proxyurl="http://"+proxyusernm+":"+proxypasswd+"@"+proxyaddr+":"+"%d"%proxyport

t1 = time.time()
r = requests.get(url,proxies={'http':proxyurl,'https':proxyurl},headers={
    "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Encoding":"gzip, deflate",
    "Accept-Language":"zh-CN,zh;q=0.9",
    "Cache-Control":"max-age=0",
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"})
r.encoding='gb2312'

t2 = time.time()

print(r.text)
print("时间差:" , (t2 - t1));
