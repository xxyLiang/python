# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymysql
import re
import traceback


class TmjyPipeline:

    def __init__(self):
        self.db = pymysql.connect(host="localhost", user="root", password="651133439a", database="rec_sys", charset='utf8mb4')
        self.cursor = self.db.cursor()
        self.sql = 'insert into posts values(' + ('%s,'*12)[:-1] + ')'

    def process_item(self, item, spider):
        # content需要去除的：\r \n \xa0
        item['content'] = re.sub('[\r\n\xa0]', '', item['content']).strip()
        if item['author_level'] == "Master":
            item['author_level'] = '11'
        try:
            self.cursor.execute(
                self.sql,
                (item['pid'], item['tid'], item['rank'], item['is_initiate_post'], item['is_thread_publisher'],
                 item['content'], item['img_num'], item['author_id'], item['author_nickname'], item['author_level'],
                 item['reply_to_pid'], item['publish_time'])
            )
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            # traceback.print_exc()
        return item
