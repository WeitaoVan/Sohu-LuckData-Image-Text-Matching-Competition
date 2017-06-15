#-*- coding: utf-8 -*-
import thulac  
import sys
reload(sys) 
sys.setdefaultencoding('utf-8')
print(sys.getdefaultencoding())
thu1 = thulac.thulac()
text = thu1.cut("我爱北京天安门", text=True)
print(text)
