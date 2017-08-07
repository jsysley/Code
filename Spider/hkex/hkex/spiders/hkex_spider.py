__author__ = 'jsysley'

import scrapy
from scrapy.http import Request
from ..items import HkexItem
from  datetime import datetime
from  datetime import timedelta


class HKEXSpider(scrapy.Spider):
    name = "hkex"
    code_list = ('00371', '00175', '03908', '00257', '00388', '01088', '00700', '02188')

    def start_requests(self):

        yield scrapy.Request(url='http://www.hkexnews.hk/sdw/search/search_sdw_c.asp',
                             meta={'code': '02188'})

    def parse(self, response):

        code = response.meta['code']
        if response.meta.has_key('date'):
            date = response.meta['date']
        else:
            date = datetime(2015, 1, 1)

        hiddenInputs = response.xpath('//*[@id="mainform"]/input[@type="hidden"]')
        formdata = {}
        formdata['sel_ShareholdingDate_d'] = str(date.day)
        formdata['sel_ShareholdingDate_m'] = str(date.month)
        formdata['sel_ShareholdingDate_y'] = str(date.year)
        formdata['txt_stock_code'] = code
        formdata['txt_stock_name'] = ''
        formdata['txt_ParticipantID'] = ''
        formdata['txt_Participant_name'] = ''

        for hidden in hiddenInputs:
            name = hidden.xpath('@name').extract()[0]
            value = hidden.xpath('@value').extract()[0]
            formdata[name] = value

        print(formdata)

        return scrapy.FormRequest("http://www.hkexnews.hk/sdw/search/search_sdw_c.asp",
                                  formdata=formdata,
                                  meta={'date': date, 'code': code},
                                  dont_filter=True,
                                  callback=self.parseData)

    def parseData(self, response):
        last_date = response.meta['date']
        code = response.meta['code']
        trs = response.xpath('//*[@id="tbl_Result_inner"]/tr[11]/td/table/tr')
        print(len(trs))
        for i in range(3, len(trs)):
            no = trs[i].xpath('td[1]/text()').extract()[0]
            if no.startswith(u'A') or no == 'B01491' or no == 'B01272' or no == 'B01451':
                unit = trs[i].xpath('td[4]/text()').extract()[0]
                percent = trs[i].xpath('td[5]/text()').extract()[0]
                item = HkexItem()
                item['code'] = code
                item['date'] = last_date
                item['NO'] = no
                item['unit'] = unit
                item['percent'] = percent
                # print item
                yield item

        next_date = last_date + timedelta(days=1)
        print(next_date)
        if next_date <= datetime.now():
            yield scrapy.Request(url='http://www.hkexnews.hk/sdw/search/search_sdw_c.asp',
                                 meta={'code': code, 'date': next_date},
                                 dont_filter=True,
                                 callback=self.parse)
