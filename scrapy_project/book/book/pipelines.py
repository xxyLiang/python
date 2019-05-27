# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymysql


class BookPipeline(object):

    # def process_item(self, item, spider):
    #     print(item["book_id"])
    #     print(item["book_name"])
    #     print(item["detail_url"])
    #     print("\n")
    #     return item

    def __init__(self):
        self.db = pymysql.connect("localhost", "root", "admin", "book", charset='utf8')
        self.cursor = self.db.cursor()
        self.book_number = 0

    def process_item(self, item, spider):
        sql = "insert into books values('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', CURRENT_TIMESTAMP)" % \
              (item["book_id"], item["b_cate"], item["s_cate"], item["book_name"], item["book_author"],
               item["book_disc"], item["book_press"], item["book_img"], item["detail_url"], 0)
        try:
            self.cursor.execute(sql)
            self.db.commit()
            self.book_number += 1
            if self.book_number % 100 == 0:
                print("have parsed %d books" % self.book_number)
        except Exception as e:
            self.db.rollback()
            print("ERROR! book %s insert to database fail" % item["book_name"])
            # print(e, '\n')
        return item

    def close_spider(self, spider):
        self.cursor.close()
        self.db.close()
        print("%d books were inserted to the database" % self.book_number)
