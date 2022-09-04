import requests
from requests.adapters import HTTPAdapter
from lxml import etree
import re
import traceback
import pymysql
import time

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()
base = "https://bbs.tnbz.com/home.php?mod=space&uid=%s&do=thread&view=me&type=reply&from=space"
referer_base = 'https://bbs.tnbz.com/home.php?mod=space&uid=%s&do=profile'
cookies = "uc_be7e_saltkey=o4fGKnCu; uc_be7e_lastvisit=1651303900; uc_be7e_pc_size_c=0; PHPSESSID=cflnos6lph6dp4nfsosg7tlbur; c=pveDt1IP-1651307503045-827cef4f9ad52393023125; _uab_collina=165130750310553718608544; Hm_lvt_e0d57af635acd52caf647bab81b97732=1651307503; Hm_lvt_05dbcd233e8cbc93a0e38dc6f0057af7=1651307503; Qs_lvt_292559=1651307503; _fmdata=QUrJgGvZnltISI5T7QPhuOj6B8VTlHJQ9ZIQ6nnX7CFLzbVMPj87gPz2dT5yfDERT4AsHh+GLEd0Y0F/CgmAy96zC38hpp2pdODViW54TOA=; uc_be7e_home_diymode=1; uc_be7e_seccode=19912.3c1d5d302e77616f33; uc_be7e_ulastactivity=94daLmMWkNIzTaWPiESJ39fpY+AKETevK0ieTOeE2Sd5XCK1jCih; uc_be7e_auth=e843f//4FvH3PIJgtX1moteRljTiB2D3ErykZguxk0Un/A8c5vaEtcAfI+INN6t0rjhXxjhQPfRmEGU4yvt93QTXHNXEGFZepfLmbYE; uc_be7e_connect_is_bind=0; uc_be7e_resendemail=1651307520; uc_be7e_nofavfid=1; uc_be7e_smile=1D1; uc_be7e_st_p=236328468164608|1651313206|58e9d577403b9ce9e35ea2328ef228b8; uc_be7e_viewid=uid_457179; uc_be7e_sid=vEaAAQ; uc_be7e_lip=116.28.49.199,1651313016; acw_tc=0bde431616513149312862995e0148f72f0a23c14e961bf634267da1eafdb6; uc_be7e_st_t=236328468164608|1651315598|b57122052336564548341bae3a65a33e; uc_be7e_forum_lastvisit=D_4_1651315598; uc_be7e_home_readfeed=1651316155; uc_be7e_checkpm=1; uc_be7e_lastcheckfeed=236328468164608|1651316155; uc_be7e_checkfollow=1; uc_be7e_lastact=1651316155	home.php	misc; uc_be7e_sendmail=1; TDpx=75; Hm_lpvt_e0d57af635acd52caf647bab81b97732=1651316157; Hm_lpvt_05dbcd233e8cbc93a0e38dc6f0057af7=1651316157; _xid=t7RbDftilI/SxiGTkbkgaqrK9wVksKOvxyZgfnngJ+8QHGswBn3kaFkwoCAHOL3EGl7NxOnrq6GrcEjto1dlxg==; Qs_pv_292559=2527062161975158000,3901523022577009000,4239741056198038500,3620115938692925400,3316605333911247400; ssxmod_itna=eq0xcQD=SxyDz6zqG=Y0QchhzXEEd+oD/9zD3q0=GFDf47f59e4D7zDm2ty0IAxosHIn0F0x9AGn7iDgWwm=ENAlD0aDbTcpCTj4YYDtxBYDQxAYDGDDP0D84DrD7r=lBfxYPG0DiK3DzTP9DGfyDYPcYqDg9qDBDD6jD7QDIT6BR8rWIeDSUYUxKi=DjLbD/RGT6r=5raakfdtiPZibYqiDxBQD7qPcDYoaIeDH8MUqcoxbRxeCROeqQoq4GRIdADmCGUh1AD3L9EK3ADAxG7G50o1kDDfmb5mxeD==; ssxmod_itna2=eq0xcQD=SxyDz6zqG=Y0QchhzXEEd4G9KkQDBqYvxDODF21BpYxDIdWGFKWschrOvheVGE8yPs9+=7H44X797yDIQP4jBnfWm/+3X62b2kKXHOXlkKhfPx4U3jBUcPayS0=7TkUrVBO6B4uLSfRvquuGsl3QHDQLF0udmn2Gx5iXhQCqgDSQKquDIBgAm3pXjWBv+jLvFqjpu38NDsPU8yo29zfNMGhE=bfXCiwERIrBK8EfXjwrn827/+LQlSa5BIg2utpvbhEMkyoBDZ+o48rYInpEGig1V8N6BsEYmGxzmsoVHePPuiCADKQR=07TuAYq0oSMselPLrYL2axQK/xKS4h2G1HQxies9qhFezuGk9k=Urh/2mcTzYhokb7RgptT++mQBlaHm3ujvHe7XDKMQkIka46stwd2okLOGS0dLrxpd+xbf=2O3mfNPKvl7oP=zaOxfpmDDwcDk7qWCxhc9V2=cRFT/FeliLewdncx05aixkYeKIyZRvKQDDjKDeuK4D=="
agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.47'
sql = 'update `users` set register_time=%s, latest_login_time=%s, last_active_time=%s, last_publish_time=%s, total_online_hours=%s where uid=%s'


s = requests.Session()
s.mount('http://', HTTPAdapter(max_retries=3))
s.mount('https://', HTTPAdapter(max_retries=3))

def get(url, ref):
    time.sleep(2)
    headers = {
        'Host': 'bbs.tnbz.com',
        'Referer': ref,
        'DNT': '1',
        'Cookie': cookies,
        'User-Agent': agent,
        'Upgrade-Insecure-Requests': '1',
        'sec-ch-ua': '" Not A;Brand";v="99", "Chromium";v="100", "Microsoft Edge";v="100"',
    }
    response = s.get(url, headers=headers, timeout=3)
    response.encoding = 'utf-8'
    return response

def parse():
    cursor.execute("select uid, first_thread_id, first_thread_time from users_df where df>7 and first_post_time is null")
    users = cursor.fetchall()

    continuous_fail = 0
    count = 0

    for user in users:
        try:
            res = get(base % user[0], referer_base % user[0])
            tree = etree.HTML(res.content)

            tree.xpath('')

            lis = tree.xpath('//ul[@id="pbbs"]/li')
            if len(lis) < 3:
                continuous_fail += 1
                continue
            for li in lis:
                try:
                    if li.xpath('.//em/text()')[0] == '在线时间':
                        total_online_hours = re.search(r'\d+', li.xpath('./text()')[0]).group()
                    elif li.xpath('.//em/text()')[0] == '注册时间':
                        register_time = li.xpath('./text()')[0]
                    elif li.xpath('.//em/text()')[0] == '最后访问':
                        latest_login_time = li.xpath('./text()')[0]
                    elif li.xpath('.//em/text()')[0] == '上次活动时间':
                        last_active_time = li.xpath('./text()')[0]
                    elif li.xpath('.//em/text()')[0] == '上次发表时间':
                        last_publish_time = li.xpath('./text()')[0]
                    else:
                        continue
                except:
                    traceback.print_exc()
            if latest_login_time is None or register_time is None:
                continuous_fail += 1
                continue
        except:
            traceback.print_exc()
            continuous_fail += 1
            continue
        try:
            cursor.execute(sql, (register_time, latest_login_time, last_active_time, last_publish_time, total_online_hours, user[0]))
            db.commit()
        except:
            db.rollback()
            traceback.print_exc()
        if continuous_fail >= 5:
            raise ConnectionError('Continuous FAIL, check cookies.')
        continuous_fail = 0
        count += 1
        print('Users %s info recorded. count: %d' % (user[0], count))


if __name__ == '__main__':
    parse()
