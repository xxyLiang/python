# coding: utf-8
from requestium import Session
from lxml import etree
import time
import datetime
import re
import traceback
import pymysql
import pickle
import random
import json


class weibo:

    def __init__(self):
        self.p = Session(webdriver_path='C:/Program Files (x86)/Google/Chrome/Application/chromedriver',
                    browser='chrome',
                    default_timeout=15, )
        self.headers = {
            'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.87 Safari/537.36"
        }
        self.db = pymysql.connect("localhost", "root", "admin", "weibo", charset='utf8mb4')
        self.cursor = self.db.cursor()
        try:
            with open("cookies2", 'rb') as f:
                self.p.cookies = pickle.load(f)
        except FileNotFoundError:
            print("Cookies Not Exist, please login first.")

    def login(self):
        try:
            self.p.driver.get('https://weibo.cn')
            time.sleep(1)
            self.p.driver.ensure_element_by_link_text("登录").click()
            time.sleep(2)
            username = self.p.driver.ensure_element_by_css_selector('#loginName')
            username.clear()
            username.send_keys('l.iangxiyiliangxiyi@gmail.com')
            time.sleep(2)
            password = self.p.driver.ensure_element_by_css_selector('#loginPassword')
            password.send_keys('651133439a')
            self.p.driver.ensure_element_by_css_selector('#loginAction').click()
            time.sleep(2)
            self.p.driver.get('https://www.weibo.com')
            time.sleep(2)
            self.p.driver.get('https://weibo.cn')
            time.sleep(3)
            self.p.transfer_driver_cookies_to_session()
            with open("cookies2", 'wb') as f:
                pickle.dump(self.p.cookies, f)
            print("登录成功")
        except():
            print('登录失败')

    # 获取某一话题下的热门微博信息
    def get_weibo_info(self, keyword):
        if len(self.p.cookies.keys()) == 0:
            print("Have not login in!!!")
            return -1
        tree = self.handle_url('https://s.weibo.com/weibo?q={}&xsort=hot&suball=1&Refer=g'.format(keyword))
        not_hot = 0
        error_list = []
        while True:
            cards = tree.xpath('//div[@class="card-wrap"][@action-type="feed_list_item"]')
            for card in cards:
                mid = card.attrib['mid']
                num_tag = card.findall('div/div[@class="card-act"]/ul/li/a')
                num = re.search(r'\d+', num_tag[1].text)
                repost = int(num.group()) if num is not None else 0
                num = re.search(r'\d+', num_tag[2].text)
                comment = int(num.group()) if num is not None else 0
                num = re.search(r'\d+', num_tag[3].xpath('string()'))
                like = int(num.group()) if num is not None else 0
                if comment < 20:
                    not_hot += 1
                    if not_hot < 5:
                        continue
                    else:
                        break

                content = card.findall('div/div[@class="card-feed"]//p[@class="txt"]')[-1].xpath('string()').strip()
                content = re.sub('收起全文d', '', content)

                t_tag = card.find('div/div[@class="card-feed"]//p[@class="from"]/a')
                send_time = self.handle_time_string(t_tag.text.strip())

                info = re.search(r'com/(\d+)/(\w+)?', t_tag.attrib['href'])
                sender = info.group(1)
                weibo_id = info.group(2)
                url = 'https://weibo.cn/comment/%s?uid=%s' % (weibo_id, sender)
                not_hot = 0

                sql = "insert into weibo_info(mid, keyword, sender, identifier_code, url, " \
                      "content, dat, reposts, comments, likes) " \
                      "values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                items = (mid, keyword, sender, weibo_id, url, content, send_time, repost, comment, like)
                try:
                    self.cursor.execute(sql, items)
                    self.db.commit()
                    print("finish getting weibo: %s / %s" % (sender, weibo_id))
                except Exception as e:
                    self.db.rollback()
                    print("Something wrong with %s / %s while getting weibo" % (sender, weibo_id))
                    error_list.append({"sender": sender, "id": weibo_id, "message": str(e)})

            nextpage = tree.xpath('//div[@class="m-page"]//a[@class="next"]')
            if not_hot >= 5 or (len(nextpage) == 0):
                break
            nextpage_url = 'https://s.weibo.com' + nextpage[0].attrib['href']
            tree = self.handle_url(nextpage_url)
        try:
            with open('log.txt', 'a') as f:
                f.writelines("\n\n%s\n###################################\n" % keyword)
                for i in error_list:
                    f.writelines("%s: %s / %s : %s\n" %
                        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i['sender'], i['id'], i['message']))
        except:
            traceback.print_exc()
        return 0

    # 获取某一页微博的评论
    def getcomment(self, url):
        if len(self.p.cookies.keys()) == 0:
            print("Have not login in!!!")
            return -1
        try:
            tree = self.handle_url(url)
            temp = tree.xpath('//div[@id="pagelist"]//input[@value="跳页"]')[0].tail
            total_page = int(re.search(r'1/(\d+)页', temp).group(1))
            if total_page > 50:
                total_page = 50
            page = 1
            while True:
                comment_tags = tree.xpath('//div[@class="c"][starts-with(@id,"C_")]')
                for tag in comment_tags:
                    reply_to = ''
                    cid = re.search(r'\d+', tag.attrib['id']).group()
                    user_page = tag.find('a').attrib['href']
                    r_time = self.handle_time_string(tag.find('span[@class="ct"]').text)
                    try:
                        like = int(re.search(r'\d+', tag.find('span[@class="cc"]/a').text).group())
                    except:
                        like = 0

                    ## 回复对象
                    ctt = tag.find('span[@class="ctt"]')
                    try:
                        if ctt.text is not None and "回复" in ctt.text and re.match('@', ctt.find('a').text) is not None:
                            reply_to = re.search(r'@(.+)', ctt.find('a').text).group(1).strip()
                    except:
                        traceback.print_exc()

                    ## 评论的表情
                    content_biaoqing = ''
                    for ele in tag.iterdescendants():
                        if (ele.tag == 'img' and 'alt' in ele.keys() and re.search(r'\[.+\]',
                                                                                   ele.attrib['alt']) is not None):
                            ## 标签是img，有alt，且alt内容满足[  ]
                            content_biaoqing += ele.attrib['alt']

                    ## 评论的文字
                    etree.strip_elements(tag, 'a', with_tail=False)  ### 开始删标签
                    for c in tag.xpath('span[@class!="ctt"]'):
                        tag.remove(c)
                    content_word = re.sub(':', '', tag.xpath('string()').replace(' ', ''), count=1)
                    content_word = re.sub('回复:', '', content_word)

                    items = [cid, page, user_page, content_word, content_biaoqing, reply_to, r_time, like]
                    yield items

                if page < total_page:
                    page += 1
                    tree = self.handle_url(url + "&page=%d" % page)
                else:
                    break
        except Exception:
            traceback.print_exc()

    def handle_url(self, url):
        try:
            res = self.p.get(url, headers=self.headers)
            res.raise_for_status()
            res.encoding = "UTF-8"
            tree = etree.HTML(res.content)
        except:
            traceback.print_exc()
            return None
        time.sleep(1)
        return tree

    @staticmethod
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

    def run_getcomment(self):
        total_count = 0
        sql = 'select `mid`,`url` from `weibo_info` where mid not in (select mid from finished_weibo)'
        self.cursor.execute(sql)
        li = self.cursor.fetchall()
        sql = "insert into comment_info(cid, page, user_page, content_words, content_bq, reply_to, dat, likes, mid) " \
              "values(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        for i in li:
            count = 0
            for items in self.getcomment(i[1]):
                items.append(i[0])
                try:
                    self.cursor.execute(sql, items)
                    self.db.commit()
                    count += 1
                    total_count += 1
                except Exception as e:
                    self.db.rollback()
            print('get comment of mid=%s have finished, %d comments are stored, %d in total.' % (i[0], count, total_count))
            try:
                self.cursor.execute('insert into finished_weibo values(%s)', [i[0]])
                self.db.commit()
            except:
                self.db.rollback()
            try:
                with open('log.txt', 'a') as f:
                    f.writelines("weibo mid=%s get %d comments\n" % (i[0], count))
            except:
                traceback.print_exc()

    def run_get_weibo(self):
        sql = 'select `keyword` from `keywords`'
        self.cursor.execute(sql)
        li = self.cursor.fetchall()
        for i in li:
            self.get_weibo_info(i[0])

    def handle_user_page(self, url):
        translate_dic = {'昵称：': 'nickname', '所在地：': 'location', '性别：': 'gender',
                         '生日：': 'birthday', '简介：': 'intro', '注册时间：': 'register_date',
                         '大学：': 'university', '高中：': 'high_school', '海外：': 'abroad',
                         }
        tree = self.handle_url(url)
        if tree is None:
            return []
        scripts = tree.xpath('script')
        info = {}
        for s in scripts:
            if s.text is None or re.match(r'FM\.view', s.text) is None:
                continue
            try:
                text = s.text.strip(';').strip(')').strip('FM.view(')
                j = json.loads(text)
                if j['domid'] == 'Pl_Core_T8CustomTriColumn__53':   # 关注 粉丝 微博量
                    s_tree = etree.HTML(j['html'])
                    table = s_tree.xpath('//table[@class="tb_counter"]//td//strong')
                    info['following'] = int(table[0].text)
                    info['follower'] = int(table[1].text)
                    info['weibo'] = int(table[2].text)
                elif j['domid'] == 'Pl_Official_PersonalInfo__57':  # 个人信息
                    s_tree = etree.HTML(j['html'])
                    cards = s_tree.xpath('//div[@class="WB_cardwrap S_bg2"]')
                    for card in cards:
                        card_name = card.find('div//h2').text
                        if card_name == "基本信息":
                            li = card.xpath('div//li')
                            for i in li:
                                if i.xpath('span')[0].text in translate_dic.keys():
                                    info[translate_dic[i.xpath('span')[0].text]] = i.xpath('span')[1].text.strip()
                        elif card_name == "教育信息":
                            li = card.xpath('div//li')
                            for i in li:
                                if i.xpath('span')[0].text in translate_dic.keys():
                                    school = i.find('span/a').text  # 注意学校是否都是链接
                                    info[translate_dic[i.find('span').text]] = school
                        elif card_name == "标签信息":
                            li = card.xpath('div//li/span[2]/a/span')
                            tag_list = []
                            for i in li:
                                tag_list.append(i.tail.strip())
                            info['tag'] = '/'.join(tag_list)
                        elif card_name == "工作信息":
                            spans = card.xpath('div//li/span[@class="pt_detail"]')
                            jobs = []
                            for s in spans:
                                try:
                                    a = s.find('a')
                                    job = a.text
                                    if a.tail is not None:
                                        job += a.tail.strip()
                                    brs = s.xpath('br')
                                    for br in brs:
                                        job += ('-' + br.tail.strip())
                                    jobs.append(job)
                                except:
                                    with open('log_user.txt', 'a') as f:
                                        f.writelines("[%s]  url %s job info error\n" %
                                                     (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), url))
                            info['job'] = '/'.join(jobs)
            except Exception as e:
                traceback.print_exc()
                with open('log_user.txt', 'a') as f:
                    f.writelines("[%s]  Something wrong when getting this url:%s, %s\n" %
                                 (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), url, str(e)))
                return []

        return info

    def run_get_user(self):
        with open('cookies', 'rb') as f:
            self.p.cookies = pickle.load(f)
        count = 0
        error_count = 0
        sql = "select distinct user_page from `comment_info` " \
              "where user_page not in (select user_page from finished_user) and user_page like'/u/%'"
        total = self.cursor.execute(sql)
        li = self.cursor.fetchall()
        for i in li:
            id = re.search(r'\d+', i[0]).group()
            url = 'https://weibo.com/' + id + '/info'
            info = self.handle_user_page(url)
            if len(info) < 5:
                info = self.handle_user_page(url)
                if len(info) < 5:
                    try:
                        self.cursor.execute("insert into error_user values(%s)", i[0])
                        self.db.commit()
                        error_count += 1
                    except:
                        self.db.rollback()
                    if error_count == 5:
                        break
                    continue
            info['id'] = id
            info['user_page'] = i[0]

            blanks = (len(info) * " %s,").strip(',')
            sql = "insert into user_info("
            item = []
            for a in info:
                sql += (a + ",")
                item.append(info[a])
            sql = sql.strip(',') + ") values(" + blanks + ")"
            try:
                self.cursor.execute(sql, item)
                self.db.commit()
                error_count = 0
                count += 1
                print("user id=%s finish, some info here: %s / %s , %d user(s) finish, %d left" %
                      (info['id'], info['nickname'], info['gender'], count, total-count))
            except Exception as e:
                self.db.rollback()
                with open('log_user.txt', 'a') as f:
                    f.writelines("[%s]  Error happened when insert id=%s into database: %s\n" %
                        (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), info['id'], str(e)))

    def transfer_id(self):
        count = 0
        sql = "SELECT distinct user_page FROM `comment_info` where user_page not like '/u/%' " \
              "and user_page not in (select user_page from error_user)"
        try:
            total = self.cursor.execute(sql)
            li = self.cursor.fetchall()
        except:
            traceback.print_exc()
            for i in li:
                tree = self.handle_url('https://weibo.cn' + i[0])
                try:
                    href = tree.xpath('//div[@class="ut"]/a[2]')[0].attrib['href']
                    id = '/u' + re.search(r'(/\d+)/info', href).group(1)
                except:
                    self.cursor.execute("insert into error_user values(%s)", i[0])
                    self.db.commit()
                    continue
                try:
                    sql2 = 'update comment_info set user_page=%s where user_page=%s'
                    self.cursor.execute(sql2, [id, i[0]])
                    self.db.commit()
                    count += 1
                    if count % 10 == 0:
                        print("finish transfer %d user ids, %d left." % (count, total-count))
                except:
                    self.db.rollback()

    def verify_info(self):
        sql = "select user_page from user_info where user_page not in (select user_page from finished_user)"
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        count = 0
        total_count = 0
        temp = []
        for r in result:
            url = "https://weibo.cn" + r[0]
            try:
                tree = self.handle_url(url)
                span = tree.xpath("//div[@class='ut']/span[@class='ctt'][1]")
                if len(span) == 0:
                    self.cursor.execute("insert into error_user values(%s)", [r[0]])
                    self.db.commit()
                    print('error span, user=%s' % r[0])
                    continue
                verify = span[0].find('img[@alt="V"]')
                member = span[0].find('a/img[@alt="M"]')
                verified = 1 if verify is not None else 0
                membered = 1 if member is not None and re.search('donate_btn_s', member.attrib['src']) is not None else 0
                temp.append((verified, membered, r[0]))
                count += 1
            except:
                traceback.print_exc()
                continue
            if count == 30:
                try:
                    for i in temp:
                        sql2 = "update user_info set verified=%s, member=%s where user_page=%s"
                        sql3 = "insert into finished_user values(%s)"
                        self.cursor.execute(sql2, i)
                        self.cursor.execute(sql3, [i[2]])
                    self.db.commit()
                except:
                    self.db.rollback()
                finally:
                    total_count += count
                    count = 0
                    temp = []
                    print("finish %d users" % total_count)


urls = [
    'https://weibo.com/6435489852/info',
    'https://weibo.com/3196470221/info',
    'https://weibo.com/3880460244/info',
    'https://weibo.com/2085387711/info',
    'https://weibo.com/5819071204/info',
    'https://weibo.com/1763709194/info',
    'https://weibo.com/1696194013/info',
    'https://weibo.com/5672469981/info',
    'https://weibo.com/3483714255/info',
    'https://weibo.com/6710935744/info',
    'https://weibo.com/1851532085/info',
    'https://weibo.com/6248697907/info',
    'https://weibo.com/5999976726/info',
]