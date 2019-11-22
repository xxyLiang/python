import requests
import json

# 医生信息包括id 姓名 性别 职称 医院 科室  tags中包含擅长领域
# 对话由"dialogs"键开始
# disease_reference 医生标注的参考疾病？

r = requests.get("https://ask.dxy.com/view/i/question/list/section?section_group_name=buxian&page_index=1")
r.raise_for_status()
r.encoding = 'utf-8'
t = json.loads(r.text)
print(r.text)