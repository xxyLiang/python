# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql


class BookPipeline(object):

    def __init__(self):
        self.db = pymysql.connect("localhost", "root", "admin", "book", charset='utf8')
        self.cursor = self.db.cursor()

    def process_item(self, item, spider):
        sql = "insert into books values('%s', '%s', '%s', '%s', '%s', '%s', '%s')" % \
              (item["book_id"], item["b_cate"], item["s_cate"], item["book_name"],
               item["book_author"], item["book_press"], item["book_img"])
        try:
            self.cursor.execute(sql)
            self.db.commit()
            print("book 《%s》 insert to database successfully" % item["book_name"])
        except:
            self.db.rollback()
            print("ERROR! book 《%s》 insert to database FAIL" % item["book_name"])
        return item

    def close_spider(self, spider):
        self.db.close()
