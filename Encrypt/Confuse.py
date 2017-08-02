#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 11:09:31 2017

@author: max
"""
import hashlib  
import random  
import os   
###############################  
# Describe : 混淆Unity脚本文件  
# D&P Author By:   常成功  
# Create Date:     2014-11-25   
# Modify Date:     2014-11-25  
###############################  
  
#想混淆的变量/方法名  
raw_name_list = ["url_open", "req", "get_page", "html","content","data"]  
                  
#混淆后的变量/方法名  
new_name_list = []  
  
#随机可选的字母表  
alphabet = ["a", "b", "c", "d", "e", "f", "g",    
    "h", "i", "j", "k", "l", "m", "n", "o", "p", "q",    
    "r", "s", "t", "u", "v", "w", "x", "y", "z"]   
  
  
#生成新的变量名  
def create_new_name() :   
    m = hashlib.md5()   
    #生成随机变量名   
    for raw_name in raw_name_list:   
        m.update(raw_name)  
        #生成一个16位的字串   
        temp_name = m.hexdigest()[0:16]  
        #合法名称校验  
        #强制以字母作为变量/方法名的开头   
        if temp_name[0].isdigit():   
            initial = random.choice(alphabet)  
            temp_name = initial + temp_name   
            temp_name = temp_name[0:16]   
        #不能重名  
        while(1):   
            if temp_name in new_name_list :  
                initial = random.choice(alphabet)  
                temp_name = initial + temp_name  
                temp_name = temp_name[0:16]   
            else:  
                new_name_list.append(temp_name)  
                break  
  
#混淆文件  
def confuse_file(root,files):    
    file_content = ""
    path_filename = os.path.join(root,files) # 带路径的文件名
    #读文件内容   
    f = file(path_filename)  
    # if no mode is specified, 'r'ead mode is assumed by default  
    while True:  
        line = f.readline()   
        if len(line) == 0: # Zero length indicates EOF  
            break   
        #混淆  
        for name_index,raw_name in enumerate(raw_name_list):   
            the_new_name = new_name_list[name_index]   
            line = line.replace(raw_name, the_new_name)
        file_content += line  
    f.close()   
    #重写文件 
    outdir_path = root + '/Encrypted' # 新建文件夹Encrypted
    if not os.path.exists(outdir_path): # 将加密的文件放在原来路径的Encrypted下
        os.makedirs(outdir_path)
    with file(os.path.join(outdir_path, files), 'w') as f:
        f.write(file_content)  
      
# 遍历当前目录下的所有.py文件      
def confuse_all(inpath):   
    #获取当前目录  
    dir_ = inpath # os.getcwd()   
    for root, dirs, filename in os.walk(dir_):    
        for files in filename:
            if files.endswith('.py'):   
                confuse_file(root,files)    
                print "Confuse File: ", files  

if __name__=="__main__":
    mode = raw_input("Code or Decode(inpput c or d):")
    inpath = raw_input('Input dir/file path:')
    if mode == 'c':
        create_new_name()
        confuse_all(inpath)   
        #打印一下混淆的情况.   
        #如果用文本保存起来, 那么以后可以反混淆, 还原文件  
        print "Start Confuse ...."
        struc = inpath + '/Encrypted/struction.txt'
        with open(struc,'w') as f:
            for j in range(0, len(raw_name_list)) :  
                print raw_name_list[j] , " --> " , new_name_list[j]
                info = raw_name_list[j] + ",--> ," + new_name_list[j]
                f.write(info + '\n')
        print "Confuse Complete !"
    elif mode == 'd':
        raw_name_list = []
        new_name_list = []
        with open(inpath + '/struction.txt','r') as f:
            while True:
                line = f.readline().strip()
                if not line:
                    break
                line = line.split(",")
                raw_name_list.append(line[2])
                new_name_list.append(line[0])
        confuse_all(inpath)   
        print "Start Deconfuse ...."
        for x1,x2 in zip(raw_name_list,new_name_list):
            print x1, " --> ", x2
        print "DeConfuse Complete !"
            