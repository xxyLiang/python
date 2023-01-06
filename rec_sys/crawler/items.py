# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class TmjyItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pid = scrapy.Field()
    tid = scrapy.Field()
    rank = scrapy.Field()
    is_initiate_post = scrapy.Field()
    is_thread_publisher = scrapy.Field()
    img_num = scrapy.Field()
    content = scrapy.Field()
    author_id = scrapy.Field()
    author_nickname = scrapy.Field()
    author_level = scrapy.Field()
    reply_to_pid = scrapy.Field()
    publish_time = scrapy.Field()
