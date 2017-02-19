# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:55:31 2016

@author: jsysley
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:59:13 2016

@author: jsysley
"""
#有道词典翻译
import urllib.request
import urllib.parse
import json
import time
import random

while True:
    content=input('请输入要翻译的内容(输入q退出程序)： ')
    if content=='q':
        break
    url='http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=https://www.baidu.com/link'
    
    
    #设置User-Agent的值实现隐藏，访问完之后可以使用response.header()来查看
    head={}
    head['User-Agent']='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36'
    
    
    
    #post方法，因此data要赋值
    data={}
    data['type']='AUTO'
    data['i']=content
    data['doctype']='json'
    data['xmlVersion']='1.8'
    data['keyfrom']='fanyi.web'
    data['ue']='UTF-8'
    data['action']='FY_BY_CLICKBUTTON'
    data['typoResult']='true'
    
    #对data编码,post方法的data对象必须这样处理
    data=urllib.parse.urlencode(data).encode('utf-8')#encode()把Unicode文件编码成其他形式
    
    #隐藏ip
    iplist=['119.166.135.37:81','182.203.2.36:8888','223.166.8.220:8888']#创建ip列表，下面随机用ip
    proxy_support=urllib.request.ProxyHandler({'http':random.choice(iplist)})
    #proxy_support=urllib.request.ProxyHandler({'http':'119.166.135.37:81'})
    opener=urllib.request.build_opener(proxy_support)#生成opener

    #urllib.request.install_opener(opener)#安装opener
     
    
    req=urllib.request.Request(url,data,head)
    response=opener.open(req)   
    #response=urllib.request.urlopen(req)
     
    html=response.read().decode('utf-8')#decode()把其他文件变成Unicode编码形式
    
    #print(html)
    
    #以上为json结构，下面载入
    #import json
    target=json.loads(html)#此时target为字典格式
    #下面访问翻译：
    print(target['translateResult'][0][0]['tgt'])
    
    time.sleep(5)#5秒