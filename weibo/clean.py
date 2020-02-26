def getcomment(url):    ## 功能要求：最好把回复的指向也弄上，评论的赞还没弄
    try:
        p.transfer_driver_cookies_to_session()
        res = handle_url(url)
        tree = etree.HTML(res.content)
        weibo_userid = tree.xpath('//div[@id="M_"]/div/a/@href')[0].split('/')[-1]
        weibo_content = ' '.join(tree.xpath('//div[@id="M_"]/div/span[@class="ctt"]/text()'))
        t = tree.xpath('//div[@id="M_"]//span[@class="ct"]/text()')[0]  # 匹配时间字符串
        weibo_time = handle_time_string(t)
        pl = tree.xpath('//span[@class="pms"]')[0]
        num = re.search(r'评论\[(\d+)\]', pl.text)
        weibo_comment = int(num.group(1)) if num is not None else 0
        num = re.search(r'转发\[(\d+)\]', pl.getprevious().find('a').text)
        weibo_repost = int(num.group(1)) if num is not None else 0
        num = re.search(r'赞\[(\d+)\]', pl.getnext().find('a').text)
        weibo_like = int(num.group(1)) if num is not None else 0
        print("%s\n%s\n评论%d 转发%d 赞%d\n"%(weibo_userid, weibo_content, weibo_comment, weibo_repost,
                                       weibo_like), weibo_time, '\n\n\n')
        page = 1
        while True:
            time.sleep(1)
            comment_tags = tree.xpath('//div[@class="c"][starts-with(@id,"C_")]')
            for tag in comment_tags:
                reply_to = ''
                temp = re.search(r'\d+', tag.find('a').attrib['href'])
                uid = temp.group() if temp is not None else re.search(r'/(.+)', tag.find('a').attrib['href']).group(1)
                r_time = handle_time_string(tag.find('span[@class="ct"]').text)
                try:
                    like = int(re.search(r'\d+', tag.find('span[@class="cc"]/a').text).group())
                except:
                    like = 0
                ctt = tag.find('span[@class="ctt"]')
                if ctt.text is not None and "回复" in ctt.text and re.match('@', ctt.find('a').text) is not None:  ##双重判断是不是回复
                    reply_to = re.search(r'@(.+)', ctt.find('a').text).group(1).strip()
                content_biaoqing = ''
                for ele in tag.iterdescendants():
                    if(ele.tag=='img' and 'alt' in ele.keys() and re.search(r'\[.+\]', ele.attrib['alt']) is not None):
                    ## 标签是img，有alt，且alt内容满足[  ]
                        content_biaoqing += ele.attrib['alt']
                etree.strip_elements(tag, 'a', with_tail=False)     ### 开始删标签
                for c in tag.xpath('span[@class!="ctt"]'):
                    tag.remove(c)
                content_word = re.sub(r':', '', tag.xpath('string()').replace(' ', ''), count=1)
                if("回复:" in content_word):
                    content_word = re.sub('回复:', '', content_word)
                print("%s\n%s%s\n%s\t赞%d\nreply_to %s\n\n"%(uid, content_word, content_biaoqing, r_time, like, reply_to))
            a = tree.xpath('//div[@id="pagelist"]/form/div/a')   ## 可能根本没有翻页按钮
            if len(a)!=0 and a[0].text == '下页' and page<50:
                url = 'https://weibo.cn' + a[0].attrib['href']
                res = handle_url(url)
                tree = etree.HTML(res.content)
                page += 1
            else:
                break
    except Exception as e:
        print(str(e))


def handle_url(url):
    try:
        res = p.get(url, headers=headers)
        res.raise_for_status()
        res.encoding = res.apparent_encoding
    except:
        traceback.print_exc()
        res = None
    return res

def handle_time_string(t):
    try:
        if "分钟前" in t:
            min = int(re.search(r'(\d+)分钟前', t).group(1))
            dt = (datetime.datetime.now() - datetime.timedelta(minutes=min)).strftime("%Y-%m-%d %H:%M")
        elif "今天" in t:
            dt = datetime.date.today().strftime("%Y-%m-%d") + re.search(r'今天(\s\d{2}:\d{2})', t).group(1)
        elif "月" in t:
            tt = re.search(r'(\d{2})月(\d{2})日(\s+\d{2}:\d{2})', t)
            dt = datetime.date.today().strftime("%Y-") + tt.group(1) + '-' + tt.group(2) + tt.group(3)
        else:
            dt = re.search(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}', t).group()
    except:
        dt = '0000-00-00 00:00'
    return dt

p.driver.get('http://weibo.cn')
time.sleep(1)
p.driver.ensure_element_by_link_text("登录").click()
time.sleep(2)
username = p.driver.ensure_element_by_css_selector('#loginName')
username.clear()
username.send_keys('416236328@qq.com')
time.sleep(2)
password = p.driver.ensure_element_by_css_selector('#loginPassword')
password.send_keys('651133439weibo')
p.driver.ensure_element_by_css_selector('#loginAction').click()
print("登录成功")
time.sleep(3)
p.transfer_driver_cookies_to_session()

for card in cards:
    content = card.findall('div[@class="card-feed"]//p[@class="txt"]')[-1].xpath('string()').strip()
    content = re.sub('收起全文d', '', content)
    print(content, '\n\n')

    t_tag = card.find('div[@class="card-feed"]//p[@class="from"]/a')
    send_time = handle_time_string(t_tag.text.strip())
    info = re.search(r'com/(\d+)/(\w+)?', t_tag.attrib['href'])
    sender = info.group(1)
    weibo_id = info.group(2)
    print("%s\n%s\n%s\n\n" % (send_time, sender, weibo_id))


p.driver.get('https://www.weibo.com')
time.sleep(2)
username = p.driver.ensure_element_by_xpath('//div[@class="W_login_form"]/div[@class="username"]/div/input')
username.clear()
username.send_keys('416236328@qq.com')

for card in cards:
    num_tag = card.findall('div[@class="card-act"]/ul/li/a')
    num = re.search(r'\d+', num_tag[1].text)
    repost = int(num.group()) if num is not None else 0
    num = re.search(r'\d+', num_tag[2].text)
    comment = int(num.group()) if num is not None else 0
    num = re.search(r'\d+', num_tag[3].xpath('string()'))
    like = int(num.group()) if num is not None else 0
    print(repost, comment, like, '\n')