# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:59:13 2016

@author: jsysley
"""
#有道词典翻译
import urllib.request
import urllib.parse
import os#创建文件夹


def url_open(url):
    req=urllib.request.Request(url)#进入网址
    #伪装客户端
    req.add_header('User-Agent','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36')
    response=urllib.request.urlopen(req)    
    html=response.read()#因为下面图片要用这个函数，图片decode的话格式不对，因此这里不decde
    
    return html

def get_page(url):
    html=url_open(url).decode('utf-8')   
    
    #req=urllib.request.Request(url)#进入网址
    #伪装客户端
    #req.add_header('User-Agent','Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.116 Safari/537.36')
    #response=urllib.request.urlopen(req)    
    #html=response.read()#因为下面图片要用这个函数，图片decode的话格式不对，因此这里不decde
    #html=response.read().decode('utf-8')#解码,html是一个字符串
    
    #下面要找到那个数字
    a=html.find('current-comment-page')+23#find返回该字符串的首字母index，偏移23刚好到数字的开头 
    b=html.find(']',a)#表示从a开始，找到’]‘，然后返回其索引坐标
    #print(html[a:b])#打印这个数字
    return html[a:b]


def find_imgs(page_url):
    html=url_open(page_url).decode('utf-8')#得到字符串的文档，以此寻找
    img_addrs=[]#给一个列表，所有图片的地址都放到这里
    
    #下面开始找每张图片的地址
    a=html.find('img src=')
    
    while a!=-1#整个页面找图    
    b=html.find('.jpg',a,a+255)#给了一个起始范围，还有结束访问，有可能没有不是jpg格式的，其他格式则为其他字符
    
    #若找不到则返回-1
    if b!=-1:
        img_addrs.append(html[a+9:b+4])#只要地址
    else:#没有找到时，b的位置应该变，否则会每次循环同一个位置，因为下面的a的下一次循环开始位置用到b
        b=a+9 
    
    #下一次开始找从上一次的结束为止开始
    a=html.find('img src',b)


    for each in img_addrs:
        print (each)
        
def save_imgs(folder,img_addrs):
    pass

def download_mm(folder="ooxx",pages=10):#文件夹名字，下载前10页
    os.mkdir(folder)#创建文件夹
    os.chdir(folder)#进入这个目录，下面下载图片就进到这个文件夹

    url="http://jandan.net/ooxx"#网址
    page_num=int(get_page(url))#得到网页地址的那个数字
    
    for i in range(pages):
        page_num-=i#拿到数字
        page_url=url+'page-'+str(page_num)+'#comments'#即打开每张图片的那个网页url
        img_addrs=find_imgs(page_url)#在每个网页拿到所有图片的地址,img_addrs是一个列表
        save_imgs(folder,img_addrs)
    
#测试模块 
if __name__=='__main__':
    download_mm()