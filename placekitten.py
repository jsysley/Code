# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 22:50:45 2016

@author: jsysley
"""

import urllib.request

req=urllib.request.Request('http://placekitten.com/')
response=urllib.request.urlopen(req)
html=response.read()#因为下面图片要用这个函数，图片decode的话格式不对，因此这里不decde
print(html)