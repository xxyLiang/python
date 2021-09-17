import requests
from requests.adapters import HTTPAdapter
from lxml import etree
import re
import traceback
import pymysql
import time

db = pymysql.connect(host="localhost", user="root", password="651133439a", database="tmjy", charset='utf8mb4')
cursor = db.cursor()
base = "https://bbs.tnbz.com/home.php?mod=space&uid=%s&do=profile"
referer_base = 'https://bbs.tnbz.com/forum.php?mod=viewthread&tid=%s'
cookies = "c=FgOuImUD-1627626809229-a67d47c720d3b-681378572; _fmdata=madm1CdIwSgAGe02IBc7JHWN2hsccac6wSoDXIi9bfEEqs0XaWtyB6%2F7NdbR5bZAwB7yjixDOUAAWO6QHtWE85LOKdyDjfzRfPfyHWSVZec%3D; _uab_collina=162762681077248288314505; vaptchaNetway=cn; Hm_lvt_b78928a5d7004f5ea572a18567e06a68=1630839350; uc_be7e_saltkey=mK6V2zWk; uc_be7e_lastvisit=1631255575; uc_be7e_pc_size_c=0; uc_be7e_auth=a1cbrUxh%2BRJ5XQ04Ba3EomHaVOnmYkAL7V93VPvJhtLYRDuv3ENu5iFdaENj6bSXOkguMGo9xNL%2FyJ1OT6FParE7SIBVfrJ34SB0O5o; uc_be7e_connect_is_bind=0; uc_be7e_smile=1D1; uc_be7e_nofavfid=1; uc_be7e_home_readfeed=1631692332; acw_tc=0bde431616317735072705010e01489be08eb632e045363b6837397bc6df62; uc_be7e_sid=yF32qS; uc_be7e_lip=218.197.153.164%2C1631718964; PHPSESSID=b7b2lv228c080bb042prh9u5rb; uc_be7e_ulastactivity=9a58sNYquiPkp7n0yVlwV3dM1R3Ljev%2BfW%2FweZbsDH0SO6I9arxp; uc_be7e_lastcheckfeed=236328468164608%7C1631773508; uc_be7e_sendmail=1; TDpx=26; uc_be7e_noticeTitle=1; Hm_lvt_e0d57af635acd52caf647bab81b97732=1631063206,1631151752,1631688745,1631773510; Hm_lvt_05dbcd233e8cbc93a0e38dc6f0057af7=1631063206,1631151752,1631688745,1631773510; Qs_lvt_292559=1631262796%2C1631497014%2C1631585102%2C1631688745%2C1631773509; uc_be7e_st_t=236328468164608%7C1631773510%7Ccfa6bf3bc9413f30d386798945ee1ecc; uc_be7e_forum_lastvisit=D_2_1631692293D_4_1631773510; uc_be7e_st_p=236328468164608%7C1631773515%7C2a068cf565ddfdfea9a5207d0490dbb3; uc_be7e_viewid=tid_238640672917504; uc_be7e_lastact=1631773523%09home.php%09space; uc_be7e_home_diymode=1; Hm_lpvt_e0d57af635acd52caf647bab81b97732=1631773526; Hm_lpvt_05dbcd233e8cbc93a0e38dc6f0057af7=1631773526; Qs_pv_292559=2497624466879500300%2C2451041726202662400%2C2203929140117382100%2C1010355115627183900%2C2187754712047489300; _xid=%2FftCuqeuFgyEGHX1f%2BN8oD%2BvitFNzkUOIhQug7JJqKlQuPc%2F0JB%2BGO201YOXDonae0Se789xfmNGMAEwyjSSPA%3D%3D; ssxmod_itna=YqGxBDyWitG=0QDXDnD+oFhaeSxfO7RLRDuLxqGX3XYGRDCqAPGfDI3bzRGYf70uxeE43OqhhaYn7B4dEC8YPhE/+DB3DExM2DQADeeDvDCeDIDWeDiDG47D=xGYDjAKzCcDm4i7DYuPDXxqkDickDm+8mxGCMxDCDGFFDQKDuEFCO6AdT4D1NeHPaAKD9aoDsg2EeAIwAf3xcP5A+3uDYDGdPGLAeQGxlbG0aC73mxYGDCKDjg+8DmeQT+kky1ANTmbYrnuXrlG9KlGxy804NGRehnh5Y7hHfT/5oYtK9srPDDpbHBaP4D=; ssxmod_itna2=YqGxBDyWitG=0QDXDnD+oFhaeSxfO7RLRDuLxA=np2KD/QQYDFEh1U3y2PAppg4uFnduj5T0=mHpxBQGkbXKuzUr3UEqznMxt8PSZQ7YUtxt0P6+qfBf+rCYCcm9w6X256OTqUi+CAIiRqVIxE/S40gIhKgSRw83l6/I4o52h3H/xR/R2ihCxxBz2TYpL4YvbUEzMWTumvom1KsfD05iAN8WMtK2xc6I83ex3bSlR6+MkXOtPupZayvTU2Pgd6+5D6yAx88YPuD2ngIzEdcY0SQMS6HifLdhs2odmTHI3aHGjEheacMvhZGWHAnvSwAjyTQpdSRc05AI+l87meppDBh772zrpRemjAV8I/39i9WYL0y4omA54qOooKZ+75hTszv2lpumv3HvxpWIWAT3WTRxDkRfKEYkHKE5yReuSmXliW2Fgz22j+8A43D0782qR0==8e2xo1ZLz4x+1qDf4D7=DYKxeD=="
agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36 Edg/93.0.961.47'
sql = 'update `users` set register_time=%s, latest_login_time=%s, last_active_time=%s, last_publish_time=%s, total_online_hours=%s where uid=%s'


def parse():
    continuous_fail = 0
    count = 0
    cursor.execute("select uid, first_thread_id from `users` where register_time is null")
    users = cursor.fetchall()
    s = requests.Session()
    s.mount('http://', HTTPAdapter(max_retries=3))
    s.mount('https://', HTTPAdapter(max_retries=3))
    for user in users:
        time.sleep(2)
        register_time = None
        latest_login_time = None
        last_active_time = None
        last_publish_time = None
        total_online_hours = None
        headers = {
            'Host': 'bbs.tnbz.com',
            'Referer': referer_base % user[1],
            'DNT': '1',
            'Cookie': cookies,
            'User-Agent': agent
        }
        try:
            response = s.get(base % user[0], headers=headers, timeout=3)
            response.raise_for_status()
            response.encoding = 'utf-8'
            tree = etree.HTML(response.content)
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
