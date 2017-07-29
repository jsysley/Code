#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 22:18:30 2017

@author: jsysley
"""
import os

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
