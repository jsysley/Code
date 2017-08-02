# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:59:13 2016

@author: jsysley
"""
#有道词典翻译
import urllib.request#获取url
import urllib.parse#解码读码
import json#target的解读需要

content=input('清输入要翻译的内容： ')
url='http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=https://www.baidu.com/link'

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

#对data编码
data=urllib.parse.urlencode(data).encode('utf-8')#encode()把Unicode文件编码成其他形式

response=urllib.request.urlopen(url,data)
 
html=response.read().decode('utf-8')#decode()把其他文件变成Unicode编码形式

#print(html)

#以上为json结构，下面载入
#import json
target=json.loads(html)#此时target为字典格式
#下面访问翻译：
print(target['translateResult'][0][0]['tgt'])