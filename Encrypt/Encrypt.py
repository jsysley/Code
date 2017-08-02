#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 20:32:41 2017

@author: jsysley
"""

### For a file
#inpath = r'/Users/jsysley/Documents/python/try.py'
#outpath = r'/Users/jsysley/Documents/try.pyc'
#import py_compile as pc
#pc.compile(inpath,outpath)

### For a dir
#inpath = r'/Users/jsysley/Documents/python'
#outpath = r'/Users/jsysley/Documents'
#import compileall
#compileall.compile_dir(inpath,outpath)

#AESKey
from Crypto.Cipher import AES
from binascii import b2a_hex, a2b_hex

obj = AES.new('c28d3dbb05c79832', AES.MODE_CBC, 'O0oO0oO0o0O0O0o0')
message = "03f3 0d0a 948d 2659 6300 0000 0000 0000"#"The answer is no"
length = 16
count = len(message)
if count < length:
    num = (length - count)
    message = message + ('\0' * num)
elif count > length:
    num = (length-(count % length))
    message = message + ('\0' * num)    
ciphertext = obj.encrypt(message)
# 转化为16进制字符
ciphertext = b2a_hex(ciphertext)


ciphertext = a2b_hex(ciphertext)
obj2 = AES.new('c28d3dbb05c79832', AES.MODE_CBC, 'O0oO0oO0o0O0O0o0')
demessage = obj2.decrypt(ciphertext)
demessage.rstrip('\0')

###rot13
tr 'A-M N-Z a-m n-z' 'N-Z A-M n-z a-m' < 1.py > t5.py # 将1.py转成t5.py(rot13编码)
