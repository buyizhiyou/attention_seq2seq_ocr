#-*-coding:utf8-*-

__author__ = "buyizhiyou"
__date__ = "2017-11-8"

import os
import random
# with open('items.txt') as f:
# 	lines = f.readlines()
# filelist = os.listdir("./sample/")
# g = open('label.txt','w')
# for f in filelist:
# 	index = int(f.split('_')[0])
# 	g.write(f+' '+lines[index])
# g.close()

# with open('label.txt','r') as f:
# 	lines = f.readlines()
# 	with open('train.txt','w') as g:
# 		g.writelines(lines[:28000])
# 	with open('val.txt','w') as h:
# 		h.writelines(lines[28000:])




with open('labels2.txt','r') as f:
	lines = f.readlines()
	random.shuffle(lines)
	with open('train.txt','w') as g:
		g.writelines(lines[:int(0.9*len(lines))])
	with open('val.txt','w') as h:
		h.writelines(lines[int(0.9*len(lines)):])
