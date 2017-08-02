# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:59:13 2016

@author: jsysley
"""
#有道词典翻译
import urllib.request
import random

#url='http://www.whatismyip.com.tw/'#一个网站，访问时就会返回你访问时的ip
url='https://s.taobao.com/search?q=%E8%8A%B1%E8%8C%B6&refpid=430268_1006&source=tbsy&style=grid&tab=all&pvid=41dfe4c9c624bbc4215e02e627ec50a7&clk1=b98d6146ca538d1897cf3f9a17632eeb&spm=a21bo.50862.201856-sline.3.5XWz4o'

#ip伪装 
iplist=['119.6.144.73:81','183.203.208.166:8118','111.1.32.28:81']#创建ip列表，下面随机用ip

proxy_support=urllib.request.ProxyHandler({'http':random.choice(iplist)})

opener=urllib.request.build_opener(proxy_support)#生成opener

#伪装访问客户端
opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36')]

#开始爬虫

#urllib.request.install_opener(opener)#安装opener

response=opener.open(url)

#response=urllib.request.urlopen(url)
html=response.read().decode('utf-8')

print(html) 