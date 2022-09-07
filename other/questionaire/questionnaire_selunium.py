import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import pandas as pd
import re


drug_dict = {"MDMA片剂（俗称“摇头丸”）": "摇头丸", "苯丙胺": "苯丙胺(安非他明)", "大麻": "大麻(麻烟)", "大麻树脂": "大麻(麻烟)",      ### ?
             "芬太尼": "芬太尼", "海洛因": "海洛因", "合成大麻素类": "大麻(麻烟)", "甲基苯丙胺（俗称“冰毒”）": "甲基苯丙胺/冰毒",
             "甲基苯丙胺片剂（俗称“麻古”、“麻果”）": "麻谷丸(冰毒片)", "甲卡西酮": "甲卡西酮", "咖啡因": "咖啡因(面面儿)",
             "可卡因": "可卡因", "氯胺酮（俗称“K粉”）": "氯胺酮(k粉)", "吗啡": "吗啡", "美沙酮": "美沙酮口服液", "七唑仑": "",        ### a
             "曲马多": "曲马多", "溴西泮": ""}   ### a

def select_box(select_element, text):
    for i in select_element.options:
        p = re.sub('[省市区]', '', i.text)
        if p in text:
            select_element.select_by_visible_text(i.text)

driver = webdriver.Ie("./IEDriverServer.exe")
driver.get("http://111.202.232.186/sso/login?service=http%3A%2F%2F111.202.232.186%2FPF%2FcasAuthUser")
driver.find_element_by_id("username").send_keys("gds44000015")
driver.find_element_by_id("password").send_keys("GDSa123456")

# 完成登录
driver.get("http://111.202.232.186/DAM/jsp/adr_dam/report/ifream.jsp")

iframe = driver.find_element_by_id("mainframe")
driver.switch_to.frame(iframe)




df = pd.read_excel('data.xlsx')

sum = df.shape[0]

i = -1
while True:
    i += 1
    if i == sum:
        break

    p = df.iloc[i]
    drug = []
    drug.append(p.iloc[14:23])

    if i < sum-1:
        while True:
            p_next = df.iloc[i+1]
            if np.isnan(p_next['性别']):
                drug.append(p_next.iloc[14:23])
                i += 1
            else:
                break

    name = driver.find_element_by_name("R_PERSON_NAME")
    name.clear()
    name.send_keys(p['姓名'])

    fill_table_time = driver.find_element_by_name("R_DT_CREATE")
    fill_table_time.clear()
    fill_table_time.send_keys(p['填表时间'].strftime('%Y-%m-%d'))

    birth_date = driver.find_element_by_name("R_DT_BIRTHDAY")
    birth_date.clear()
    birth_date.send_keys(p['出生日期'].strftime('%Y-%m-%d'))

    huji = p['户籍地详址']
    huji_province_select = Select(driver.find_element_by_id("DOMICILE_PROVINCE"))
    select_box(huji_province_select, huji)
    huji_city_select = Select(driver.find_element_by_id("R_DOMICILE_CITY"))
    select_box(huji_city_select, huji)
    huji_county_select = Select(driver.find_element_by_id("R_DOMICILE_COUNTY"))
    select_box(huji_county_select, huji)

    living = p['居住地详址']
    living_province_select = Select(driver.find_element_by_id("LIVING_PROVINCE"))
    select_box(living_province_select, living)
    living_city_select = Select(driver.find_element_by_id("R_LIVING_CITY"))
    select_box(living_city_select, living)
    living_county_select = Select(driver.find_element_by_id("R_LIVING_COUNTY"))
    select_box(living_county_select, living)

    gender = p['性别']
    if gender == '男' or gender == '女':
        gender_select = Select(driver.find_element_by_id("R_SEX"))
        gender_select.select_by_visible_text(gender)

    nation = re.sub('族', '', p['民族'])
    try:
        nation_select = Select(driver.find_element_by_id("R_NATION"))
        nation_select.select_by_visible_text(nation)
    except:
        pass

    try:
        marry = p['婚姻状况']
        marry_select = Select(driver.find_element_by_id("R_MARRY"))
        if marry in ['已婚', '复婚', '再婚']:
            marry_select.select_by_visible_text('已婚(含再婚)')
        elif marry == '未婚':
            marry_select.select_by_visible_text('未婚')
        elif marry == '丧偶':
            marry_select.select_by_visible_text('丧偶')
        elif marry == '离婚':
            marry_select.select_by_visible_text('离婚')
        else:
            pass
    except:
        pass

    try:
        job = p['从业状况']
        job_select = Select(driver.find_element_by_id("R_JOB_STATUS"))
        if job == '个体经营户':
            job_select.select_by_visible_text('个体经营')
        elif job == '工人':
            job_select.select_by_visible_text('企/事业职工(工人)')
        elif job == '农民':
            job_select.select_by_visible_text('农民')
        elif job == '其它':
            job_select.select_by_visible_text('其他')
        elif job == '企业管理人员':
            job_select.select_by_visible_text('')           ####！！！！
        elif job == '退（离）休人员':
            job_select.select_by_visible_text('离/退休人员')
        elif job == '无业人员':
            job_select.select_by_visible_text('无业')
        elif job == '职员':
            job_select.select_by_visible_text('')           #### ！！！！
        elif job == '自由职业者':
            job_select.select_by_visible_text('自由职业者')
        else:
            pass
    except:
        pass

    try:
        degree = p['文化程度']
        degree_select = Select(driver.find_element_by_id("R_DEGREE"))
        if re.search('研究生', degree) is not None:
            degree_select.select_by_visible_text('大学以上')
        elif re.search('大学本科', degree) is not None:
            degree_select.select_by_visible_text('大学')
        elif degree in ['普通高级中学教育', '']:
            degree_select.select_by_visible_text('')
    except:
        pass

    drugs = p['过去12个月曾经滥用毒品种类'].split('/')
    for d in drugs:
        try:
            a = driver.find_element_by_id(drug_dict[d])
            driver.execute_script("arguments[0].click();", a)
        except:
            pass
    c = 1
    for d in drugs:
        try:
            a = driver.find_element_by_xpath("//select[@name='R_MAIN_ABUSE_DRUGCODE'][%d]" % c)
            drug_select = Select(a)
            drug_select.select_by_visible_text(drug_dict[d['主要滥用毒品种类']])