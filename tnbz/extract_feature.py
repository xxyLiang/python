import numpy as np
from bert_serving.client import BertClient
import math
import re
import paddlehub as hub
import pandas as pd
import traceback
from lda import LDA

''' Function: 将句子转化为BERT向量，提取句子特征 '''

# start server
# bert-serving-start -model_dir chinese_roberta_wwm_ext_L-12_H-768_A-12 -num_worker=1 -port=5206 -port_out=5207


class Feature:

    def __init__(self):
        print("Connecting BERT Service...")
        self.bc = BertClient(port=5206, port_out=5207)
        print("Initiating Sentiment Analysis Module...")
        self.senta = hub.Module(name="senta_lstm")
        self.positive_words = []
        self.negative_words = []
        print("Loading Sentiment Dictionary...")
        assert self.load_sentiment_dict() == 0
        self.lda_model = LDA()
        print("Initiation Complete.")
        self.lang_feature = ['sentence_length', 'numeric_words', 'pronouns_per', 'pronouns_quest',
                             'pronouns_demon', 'pronouns_neg', 'time_words', 'food_words', 'region_words', 'drug_words',
                             'diabetes_words', 'general_words', 'sentiment', 'positive_words', 'negative_words']
        self.lang_feature.extend('topic%d' % i for i in range(20))
        self.re_lang_feature = ['re_%s' % i for i in self.lang_feature]
        self.column_name = ['feature%d' % i for i in range(768)]
        self.column_name.extend(self.lang_feature)

    @staticmethod
    def count_words(sentence, types):
        types_list = {"Personal": ["他", "她", "它", "我", "你", "您", "咱", "俺", "大家", "自己", "别人"],
                      "Interrogative": ["谁", "什么", "哪", "几", "多少", "怎样", "怎么", "啥", "想知道", "是不是", "是否",
                                        "吗", "么", "会不会", "为何", "请教", "请问", "求", "提问", "？", "多久", "多快", "多远",
                                        "多大", "如何", "行不行", "能不能", "能否", "可不可以", "可否", "求助", "疑惑", "疑问",
                                        "疑虑", "知不知道", "问一下", "问下", "追问", "有没有", "有木有", "几时", "咨询", "询问", "恳求",
                                        "帮帮我", "问问", "问大家", "难道"],
                      "Demonstrative": ["这", "那", "别的", "其他", "其余", "某", "另"],
                      "Negative": ["不", "没", "无", "莫", "并非"],
                      "time": ["前天", "昨天", "今天", "明天", "后天", "上周", "这周", "下周", "日", "月", "年",
                               "早上", "晨", "上午", "中午", "下午", "晚上", "白天", "夜里", "夜晚", "今晚", "半夜", "昨晚", "睡前",
                               "早餐", "早点", "午餐", "午饭", "晚餐", "晚饭", "宵夜", "夜宵", "时间",
                               "周一", "周二", "周三", "周四", "周五", "周六", "周末", "星期", "礼拜", "一周", "两周", "三周", "每周", "几周",
                               "半天", "一天", "两天", "三天", "几天", "每天", "天天", "第二天", "小时", "个钟", "分钟", "秒", "点多",
                               "一点,", "两点", "三点", "四点", "五点", "六点", "七点", "八点", "九点", "十点", "十一点", "十二点", "点钟"
                               "以前", "现在", "未来", "将来"],
                      'food': ['吃', '米饭', '豆角', '面包', '谷类', '谷物', '五谷', '肉', '蛋', '奶', '蔬菜', '水果', '菜',
                               '黄瓜', '西红柿', '大豆', '豆制品', '面食', '饼', '主食', '小麦', '馒头', '花卷', '蒜', '薯',
                               '鸡', '鸭', '鱼', '苦瓜', '饺子', '馄饨', '耗油', '花生油', '汤', '排骨', '海带', '萝卜', '苹果',
                               '香蕉', '笋', '豆浆', '杂粮', '虾', '菇', '冬瓜', '椒', '木耳', '丝瓜', '橘子', '橙子', '西瓜',
                               '桃', '燕麦', '荞麦', '花生', '大米', '小米', '粥', '红枣', '玉米', '土豆', '菠萝', '豆腐', '扁豆',
                               '番茄', '茄子', '魔芋', '芋头', '红米', '南瓜', '糯米', '粗粮', '凉拌', '粉', '荤', '水饺', '芝麻',
                               '黄豆', '红豆', '黑豆', '绿豆']
                      }
        count = 1
        for p in types_list[types]:
            count += len(re.findall(p, sentence))
        return math.log2(count)

    @staticmethod
    def count_region_words(sentence):
        cities = ["广东", "北京", "上海", "深圳", "广州", "重庆", "天津", "江苏", "武汉", "杭州", "浙江", "湖北", "河南",
                  "湖南", "福建", "香港", "澳门", "南京", "苏州", "扬州", "厦门", "青岛", "济南", "石家庄", "哈尔滨", "郑州"
                  "四川", "河北", "山西", "辽宁", "吉林", "黑龙江", "西安", "长沙", "桂林", "南宁", "昆明", "县里", "市里",
                  "安徽", "江西", "山东", "海南", "贵州", "云南", "陕西", "甘肃", "青海", "台湾"]
        for city in cities:
            if city in sentence:
                return 1
        return 0

    @staticmethod
    def medical_words(sentence):
        # including drugs, disease, etc.
        drugs = ["胰岛素", "门冬", "地特", "列酮", "德谷", "甘精", "诺和灵", "诺和锐", "诺和达", "来得时", "长秀霖", "甘舒霖", "优思灵", "优泌乐",
                 "二甲", "双胍", "格华止", "美迪康", "捷诺达", "宜合瑞", "卡双平",               # 二甲双胍类
                 "磺酰脲", "磺脲", "格列", "美吡达", "瑞易宁", "迪沙", "依吡达", "优哒灵", "达美康", "优降糖", "克糖利", "亚莫利", "糖适平",
                 "糖100", "阿卡波", "拜糖平", "倍欣", "卡博平",                                 # α-糖苷酶抑制剂
                 "利拉鲁肽", "艾塞那肽", "普兰林肽", "度易达", "度拉糖", "百泌达",                # GLP-1
                 "卡司平", "捷诺维", "安达唐", "安立泽", "欧唐静", "尼欣那", "怡可安", "他汀", "孚来迪"]
        diabetes = ["糖耐", "OGTT", "ogtt", "Ogtt", "糖前", "空腹", "餐前", "餐后", "餐一", "餐二", "餐三", "餐1", "餐2", "餐3",
                    "1型", "一型", "2型", "二型", "dm", "DM", "Dm",  "血糖", "尿糖", "碳水", "降糖", "升糖", "生化", "紫胖", "裸奔",
                    "黄昏现象", "黎明现象", "控糖"]
        general = ["胰脏", "胰腺", "胰岛", "病变", "糖尿病足", "眼底", "视网膜", "肾", "肝", "心肌", "心脏", "神经", "血管", "胃",
                   "血浆", "血液", "血清", "血红蛋白", "血脂", "激素", "细胞", "代谢", "基因", "遗传", "抵抗", "免疫", "慢性", "急性",
                   "抗原", "抗体", "酶", "酮", "蛋白", "氨基酸", "肽", "脂肪", "肢端肥大", "库欣", "甲亢", "血压", "冠心病", "坏疽",
                   "卒中", "中风", "下肢", "便秘", "心动过速", "失禁", "酸中毒", "病因", "发病", "机制", "阳性", "阴性", "果糖胺",
                   "感染", "恶心", "呕吐", "食欲下降", "腹痛", "腹泻", "并发症", "分泌", "葡萄糖", "胰高糖素", "真菌", "细菌",
                   "电解质", "血气分析", "尿检", "诊断", "静脉", "动脉", "肥胖", "家族史", "拮抗", "应激", "糖皮质", "甲状腺",
                   "肿瘤", "胰高血糖素", "嘴", "皮肤", "口腔", "四肢", "多饮", "多尿", "多食", "消瘦", "验血", "核酸", "呼吸",
                   "头晕", "病毒", "肺", "支气管", "痰", "脱水", "尿频", "尿急", "发炎", "炎症", "炎性", "因子", "疹", "牙",
                   "内分泌", "护理", "耳鼻喉", "内科", "泌尿", "病理", "糖原", "维生素", "维他命", "尿常规", "血常规"]
        drug_words = diabetes_words = general_words = 1
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

    def load_sentiment_dict(self):
        self.positive_words.clear()
        self.negative_words.clear()
        try:
            with open("./material/NTUSD_positive_simplified.txt", 'r', encoding='utf-16') as f:
                for w in f.readlines():
                    self.positive_words.append(w.strip('\n'))
            with open("./material/NTUSD_negative_simplified.txt", 'r', encoding='utf-16') as f:
                for w in f.readlines():
                    self.negative_words.append(w.strip('\n'))
        except FileNotFoundError:
            return -1
        return 0

    def sentiment_analysis(self, sentence: str) -> float:
        assert isinstance(sentence, str)
        r = self.senta.sentiment_classify(texts=[sentence])
        return r[0]['positive_probs']

    def count_sentiment_words(self, sentence):
        p = 1
        n = 1
        for w in self.positive_words:
            if w in sentence:
                p += 1
        for w in self.negative_words:
            if w in sentence:
                n += 1
        return math.log2(p), math.log2(n)

    def get_lang_feature(self, sentence: str):
        if len(sentence) == 0:
            return np.zeros(len(self.lang_feature))
        sentence_length = math.log10(len(sentence) + 1)
        numeric_words = math.log2(len(re.findall(r'\d+\.?\d*', sentence)) + 1)
        pronouns_per = self.count_words(sentence, "Personal")
        pronouns_quest = self.count_words(sentence, "Interrogative")
        pronouns_demon = self.count_words(sentence, "Demonstrative")
        pronouns_neg = self.count_words(sentence, "Negative")
        time_words = self.count_words(sentence, "time")
        food_words = self.count_words(sentence, "food")
        region_words = self.count_region_words(sentence)  # binary
        drug_words, diabetes_words, general_words = self.medical_words(sentence)
        sentiment = self.sentiment_analysis(sentence)
        positive_words, negative_words = self.count_sentiment_words(sentence)
        features = [sentence_length, numeric_words, pronouns_per, pronouns_quest, pronouns_demon,
                    pronouns_neg, time_words, food_words, region_words, drug_words, diabetes_words,
                    general_words, sentiment, positive_words, negative_words]
        features.extend(self.lda_model.lda_feature(sentence))
        return features

    def joint_feature(self, sentences, bert_feature=True):

        language_feature = []
        for sentence in sentences:
            language_feature.append(self.get_lang_feature(sentence))
        if bert_feature:
            f = self.bc.encode(sentences)                              # embedding
            f = np.hstack([f, np.array(language_feature)])
            df = pd.DataFrame(f, columns=self.column_name)
        else:
            df = pd.DataFrame(np.array(language_feature), columns=self.re_lang_feature)
        return df

    def word2vec_feature(self, sentences, word2vec_feature=True):
        language_feature = []
        pass

