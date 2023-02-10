import random
import requests
from requests.adapters import HTTPAdapter
from lxml import etree
import re
import time
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, INTEGER, SmallInteger, CHAR, DATETIME, String, Date
from sqlalchemy.orm import sessionmaker
import traceback

engine = create_engine("mysql+pymysql://root:651133439a@localhost/rec_sys")
Base = declarative_base()
DbSession = sessionmaker(bind=engine)
sql_session = DbSession()


class Users(Base):
    __tablename__ = 'user_list'

    uid = Column(CHAR, primary_key=True)
    thread_cnt = Column(INTEGER)
    post_cnt = Column(INTEGER)
    level = Column(SmallInteger)
    user_group = Column(SmallInteger)
    total_online_hours = Column(INTEGER)
    regis_time = Column(DATETIME)
    latest_login_time = Column(DATETIME)
    latest_active_time = Column(DATETIME)
    latest_pub_time = Column(DATETIME)
    prestige = Column(SmallInteger)
    points = Column(INTEGER)
    wealth = Column(INTEGER)
    visitors = Column(INTEGER)
    friends = Column(SmallInteger)
    records = Column(SmallInteger)
    logs = Column(SmallInteger)
    albums = Column(SmallInteger)
    total_posts = Column(INTEGER)
    total_threads = Column(INTEGER)
    shares = Column(INTEGER)
    diabetes_type = Column(String)
    treatment_type = Column(String)
    gender = Column(CHAR)
    birthdate = Column(Date)
    habitation = Column(String)


class Parser:
    def __init__(self):
        self.task_cnt = 100
        self.success_cnt = 0
        self.fail_cnt = 0
        self.network_continuous_fail = 0
        self.base = "https://bbs.tnbz.com/home.php?mod=space&uid=%s&do=profile"
        self.referer_base = 'https://bbs.tnbz.com/forum.php?mod=viewthread&tid=%s'
        self.cookies = "acw_tc=0b32807e16722929122764508e0b556f92c76aefc0ae8caca2fd42d0c7555b; uc_be7e_saltkey=WaDe9MA5; uc_be7e_lastvisit=1672289312; uc_be7e_sid=KvpPfR; uc_be7e_home_diymode=1; uc_be7e_pc_size_c=0; PHPSESSID=cqfdlldbdbunad0gpq33caqprp; uc_be7e_lastact=1672292912	home.php	misc; uc_be7e_sendmail=1; c=SuHFuojh-1672292913041-4ac7da828837b-1023356641; _fmdata=vAeIHsBsqkToU4uWO8BtH6SuQnTsheZjNClQEKUKKLbL3bMCGgRNKqWyG2XHTfFXx5tJwXkHJHSnN6if1B/zvQ==; _xid=6QpHvnQq56qJT54KUea7LhNdgqxZc07Tc64UKm+n4LQ=; Hm_lvt_e0d57af635acd52caf647bab81b97732=1672292913; Hm_lpvt_e0d57af635acd52caf647bab81b97732=1672292913; Hm_lvt_05dbcd233e8cbc93a0e38dc6f0057af7=1672292913; Hm_lpvt_05dbcd233e8cbc93a0e38dc6f0057af7=1672292913; Qs_lvt_292559=1672292913; Qs_pv_292559=1724541192309466600; ssxmod_itna=mq0xR7G=D=eWwqBc4eTmIDyGKIxcDDK+0dGOFUDBwb4AQDyD8xA3GE+mk3P=CKuASvEYKAQ0xdqIbbO2+GWm9CEfzPwDB3DExumhYe+DYYDtxBYDQxAYDGDDpRDj4ibDY+/7Djnz/Zl61IdD7eDXxGCDQIdx4DaDGakqmrYDnWk96chUmYDnO+1bcKD9x0CDlpqvKbkDD5ffK4s9xfWDm+3va7hDCKDj2Y10mYDUDvsB=I6P6GaE0w7=O4PeQ04h74qH7Z4IW0bEi6B1v=7FW4SDDPPh7hg4D===; ssxmod_itna2=mq0xR7G=D=eWwqBc4eTmIDyGKIxcDDK+0dGOFD8dPOxGXheqGaRDvr61qvx8oO9dzZFQqntsPPSw4N3XQ2QSd3QLzmFN3T1EcvEQtQU7ebzXnKpKhUyAh/E226oQwHIUYjNkiz8vZkXwSrNDoWjLQYizSw0VBdXfBh=5yqX3g2h9Ki0i8TjokdficS2dgm2UD3uUDcY38TiT+co7Z3Wv4Rp51YwOsC9+Cbrhc1WwYzXHRiiPLqiWr9P3mWy68zSa6FkjzlPN0r8TBeAHFGKINfxwDCZnaWLfYjG2Q2Tz7Z5HEQhvcZ5IaLUmxlZNkG=DfQaa0xcdIlXc=jUjG+nqU7rGBpxOXZAo4P0H6oPfPitY3BfeYftcI5cm9Cp8nPtrqN9m0r=aovW/wOiwKE=nDmAQA+hI+CpO2T8trgBRzmUOjTUgEzB3N4aC/nrFbBxnx+iiSrHhuhGhH=d27r1p2h4LnKPlmFsS+3/IWDG28bK0q5oi4f+oQLzgkaldG=MKD08DijiYD==="
        self.agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36 Edg/108.0.1462.54"
        self.session = requests.Session()
        self.session.mount('http://', HTTPAdapter(max_retries=3))
        self.session.mount('https://', HTTPAdapter(max_retries=3))
        self.session.headers.update({
            'Host': 'bbs.tnbz.com',
            'DNT': '1',
            'User-Agent': self.agent
        })
        self.group_map = {"学习班": 0, "游客": 0, "未晋级": 1, "托儿所": 2, "幼儿园": 3,
                          "小学": 4, "初中": 5, "高中": 6, "大专": 7, "预科": 8, "本科": 9,
                          "硕士": 10, "博士": 11, "特邀嘉宾": 15, "特邀贵宾": 15,
                          "荣誉版主": 15, "甜蜜义工": 15, "管理员": 20, "超级版主": 20,
                          "版主": 20, "查看员": 20, "AA": 20, "管委会": 20}

    def run(self):
        users = sql_session.query(Users)\
                    .filter(Users.thread_cnt > 5)\
                    .filter(Users.regis_time.is_(None))\
                    .all()
        self.task_cnt = len(users)
        for user in users:
            self.get_user_info(user)
            time.sleep(2)

    def get_response(self, url):
        try:
            response = self.session.get(url, timeout=3)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return response
        except:
            return None

    def fail_check(self):
        self.fail_cnt += 1
        self.network_continuous_fail += 1
        if self.network_continuous_fail >= 5:
            raise ConnectionError('Continuous FAIL, check cookies.')

    def get_user_info(self, user, update=True):
        response = self.get_response(self.base % user.uid)
        if response is None:
            self.fail_check()
            return -1
        tree = etree.HTML(response.content)

        self.active_block(user, tree)
        self.info_block(user, tree)
        self.statis_block(user, tree)

        if user.regis_time is None:
            self.fail_check()
            return -1

        if update:
            try:
                sql_session.add(user)
                sql_session.commit()
                self.success_cnt += 1
            except:
                sql_session.rollback()
                traceback.print_exc()
                self.fail_cnt += 1

        self.print_progress(user)
        self.network_continuous_fail = 0
        return 1

    def active_block(self, user: Users, tree):
        active_lis = tree.xpath('//ul[@id="pbbs"]/li')
        for li in active_lis:
            try:
                tag = li.xpath('.//em/text()')[0]
                if tag == '在线时间':
                    user.total_online_hours = int(re.search(r'\d+', li.xpath('./text()')[0]).group())
                elif tag == '注册时间':
                    user.regis_time = li.xpath('./text()')[0]
                elif tag == '最后访问':
                    user.latest_login_time = li.xpath('./text()')[0]
                elif tag == '上次活动时间':
                    user.latest_active_time = li.xpath('./text()')[0]
                elif tag == '上次发表时间':
                    user.latest_pub_time = li.xpath('./text()')[0]
                else:
                    continue
            except:
                pass

        try:
            user_group_ul = tree.xpath('//ul[@id="pbbs"]/preceding-sibling::ul')
            for ul in user_group_ul:
                if re.match('用户组', ul.xpath('./li/em/text()')[0]):
                    group_str = ul.xpath('.//a//text()')[0]
                    if group_str in self.group_map:
                        user.user_group = self.group_map[group_str]
        except:
            pass

        if user.latest_login_time is None or user.regis_time is None:
            self.fail_check()
            return -1
        return 0

    def info_block(self, user: Users, tree):
        try:
            user.visitors = int(tree.xpath(".//ul[@class='pf_l cl pbm mbm']/li[1]/*[2]/text()")[0])
        except:
            pass

        statis_a = tree.xpath(".//ul[@class='cl bbda pbm mbm']//a/text()")
        for a in statis_a:
            try:
                k = a[:3]
                v = int(re.findall(r'\d+', a)[0])
                if k == '好友数':
                    user.friends = v
                elif k == '记录数':
                    user.records = v
                elif k == '日志数':
                    user.logs = v
                elif k == '相册数':
                    user.albums = v
                elif k == '回帖数':
                    user.total_posts = v
                elif k == '主题数':
                    user.total_threads = v
                elif k == '分享数':
                    user.shares = v
                else:
                    continue
            except:
                pass

        try:
            member_info_texts = tree.xpath(".//div[@class='pbm mbm bbda c']//text()")
            level = re.search(r'\[LV.(\d+|Master)]', ''.join(member_info_texts)).group(1)
            if level == "Master":
                user.level = 11
            else:
                user.level = int(level)
        except:
            pass

        personal_info_lis = tree.xpath(".//ul[@class='pf_l cl']/li")
        for li in personal_info_lis:
            try:
                tag = li.xpath("./em/text()")[0]
                value = li.xpath("./text()")[0]
                if tag == '糖尿病类型':
                    user.diabetes_type = value
                elif tag == '治疗方案':
                    user.treatment_type = value
                elif tag == '性别':
                    user.gender = value
                elif tag == '生日':
                    if value == '-' or '年' not in value:
                        continue
                    if '月' in value and '日' in value:
                        value = value.replace(' ', '').replace('日', '').replace('年', '-').replace('月', '-')
                        user.birthdate = value
                    else:
                        user.birthdate = value[:4] + '-1-1'
                elif tag == '居住地':
                    user.habitation = value
                else:
                    continue
            except:
                pass

    def statis_block(self, user: Users, tree):
        lis = tree.xpath(".//div[@id='psts']//li")
        for li in lis:
            try:
                tag = li.xpath('./em/text()')[0]
                value = li.xpath('./text()')[0]
                if tag == '积分':
                    user.points = int(value)
                elif tag == '威望':
                    user.prestige = int(value)
                elif tag == '金钱':
                    user.wealth = int(value)
                else:
                    continue
            except:
                pass

    @staticmethod
    def print_user(user: Users):
        print('uid: %s' % user.uid)
        print('用户组：%s, 签到等级：%s, 在线时间：%s 小时' % (user.user_group, user.level, user.total_online_hours))
        print('回帖数: %s, 主题数: %s, 好友数: %s, 记录数: %s, 日志数: %s, 相册数: %s, 分享数: %s' %
              (user.total_posts, user.total_threads, user.friends, user.records, user.logs,
               user.albums, user.shares))

    def print_progress(self, user: Users):
        print("\r", end="")
        self.print_user(user)
        done = int((self.success_cnt+self.fail_cnt)/self.task_cnt*50)
        undone = 50 - done
        pc = int((self.success_cnt+self.fail_cnt)/self.task_cnt*100)
        print("[{done}{undone}]({p}%) success {s} | fail {f} | total {t}"
              .format(done=">"*done, undone=" "*undone, p=pc, s=self.success_cnt, f=self.fail_cnt, t=self.task_cnt),
              end='')


if __name__ == "__main__":
    p = Parser()
    p.run()