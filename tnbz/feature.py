import numpy as np
from bert_serving.client import BertClient
import math
import re

# start server
# bert-serving-start -model_dir chinese_L-12_H-768_A-12 -num_worker=2 -port=5206 -port_out=5207 -max_seq_len=128
bc = BertClient(port=5206, port_out=5207)


def count_words(sen, types):
    pronouns_list = {"Personal": ["他", "她", "它", "我", "你", "您", "咱"],
                     "Interrogative": ["谁", "什么", "哪", "几", "多少", "怎样", "怎么", "啥"],
                     "Question": ["想知道", "是不是", "是否", "吗", "会不会", "为什么", "为何", "请教", "能否", "求解", "？"],
                     "Demonstrative": ["这", "那", "别的", "其他", "其余"],
                     "Negative": ["不", "没", "无", "莫"]
                     }
    count = 1
    for p in pronouns_list[types]:
        count += len(re.findall(p, sen))
    return math.log2(count)


def count_region_words(sen):
    cities = ["广东", "北京", "上海", "深圳", "广州", "重庆", "天津", "江苏", "武汉", "杭州", "浙江", "湖北", "河南",
              "湖南", "福建", "香港", "澳门", "南京", "苏州", "扬州", "厦门", "青岛", "济南", "石家庄", "哈尔滨", "郑州"
              "四川", "河北", "山西", "辽宁", "吉林", "黑龙江", "西安", "长沙",
              "安徽", "江西", "山东", "海南", "贵州", "云南", "陕西", "甘肃", "青海", "台湾"]
    for city in cities:
        if city in sen:
            return 1
    return 0


def medical_words(sentence):
    # including drugs, disease, etc.
    drugs = ["胰岛素", "门冬", "地特", "列酮", "德谷", "甘精", "诺和灵", "诺和锐", "诺和达", "来得时", "长秀霖", "甘舒霖", "优思灵", "优泌乐",
             "二甲", "双胍", "格华止", "美迪康", "捷诺达", "宜合瑞", "卡双平",               # 二甲双胍类
             "磺酰脲", "磺脲", "格列", "美吡达", "瑞易宁", "迪沙", "依吡达", "优哒灵", "达美康", "优降糖", "克糖利", "亚莫利", "糖适平",
             "糖100", "阿卡波", "拜糖平", "倍欣", "卡博平",                                 # α-糖苷酶抑制剂
             "利拉鲁肽", "艾塞那肽", "普兰林肽", "度易达", "度拉糖",                         # GLP-1
             "卡司平", "捷诺维", "安达唐", "安立泽", "欧唐静", "尼欣那", "怡可安", "他汀",]
    diabetes = ["糖耐", "OGTT", "ogtt", "糖前", "空腹", "餐前", "餐后", "餐一", "餐二", "餐三", "餐1", "餐2", "餐3",
                "1型", "一型", "2型", "二型", "dm", "DM", "血糖", "尿糖", "碳水", "降糖"]
    general = ["胰脏", "胰腺", "胰岛", "病变", "糖尿病足", "眼底", "视网膜", "肾", "肝", "心肌", "神经", "血管", "胃旁路手术",
               "血浆", "血液", "血清", "血红蛋白", "血脂", "激素", "细胞", "代谢", "基因", "遗传", "抵抗", "免疫", "慢性", "急性",
               "抗原", "抗体", "酶", "酮", "蛋白", "氨基酸", "肽", "脂肪", "肢端肥大", "库欣", "甲亢", "血压", "冠心病", "坏疽",
               "卒中", "中风", "下肢", "便秘", "心动过速", "失禁", "酸中毒", "病因", "发病", "机制", "阳性", "阴性", "果糖胺",
               "感染", "恶心", "呕吐", "食欲下降", "腹痛", "腹泻", "并发症", "分泌", "葡萄糖", "胰高糖素", "真菌", "细菌",
               "电解质", "血气分析", "尿检", "诊断", "静脉", "动脉", "肥胖", "家族史", "拮抗", "应激", "糖皮质", "甲状腺",
               "肿瘤", "胰高血糖素"]
    drug_words = 1
    diabetes_words = 1
    general_words = 1
    for w in drugs:
        drug_words += len(re.findall(w, sentence))
    for w in diabetes:
        diabetes_words += len(re.findall(w, sentence))
    for w in general:
        general_words += len(re.findall(w, sentence))
    drug_words = math.log2(drug_words)
    diabetes_words = math.log2(diabetes_words)
    general_words = math.log2(general_words)
    return drug_words, diabetes_words, general_words


def joint_feature(sen, init, rep):
    f = bc.encode([sen]).flatten()                              # embedding
    words_log = math.log2(len(sen) + 1)
    numeric_log = math.log2(len(re.findall(r'\d+\.?\d*', sen)) + 1)
    pronouns_per = count_words(sen, "Personal")
    pronouns_quest = count_words(sen, "Interrogative")
    pronouns_demon = count_words(sen, "Demonstrative")
    neg_words = count_words(sen, "Negative")
    region_words = count_region_words(sen)                      # binary
    drug_words, diabetes_words, general_words = medical_words(sen)

    f = np.append(f, [...])
    return f

