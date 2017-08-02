# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy

class HkexItem(scrapy.Item):
    code = scrapy.Field() # 股票代码
    NO = scrapy.Field()   # 席位编号
    date = scrapy.Field() # 数据日期
    unit = scrapy.Field() # 持股数
    percent = scrapy.Field() # 比例
