#-*-coding:utf8-*-

__author ="buyizhiyou"
__date ="2018-7-26"

'''
generate data for training
'''
import sys,os,pdb
import numpy as np 
import random
import re

from PIL import Image
from PIL import ImageFont,ImageDraw,ImageFilter
from skimage.util  import random_noise
from skimage import io,transform

def get_len(line):
    '''
	return length of line,we regard one chinese char as 1 but one number and english char
	as 50% length compared with chinese char
	'''
    chinese_chars = re.findall(u'[\u4e00-\u9fa5]', line)  # chinese chars
    chinese_length = len(chinese_chars)  # length of chinese chars
    rest_leng = int(0.5*(len(line) - chinese_length)) # length of english chars,numbers and others

    length = chinese_length + rest_leng
    # print(length)

    return length


def gen_image(h,i,text,length):
    '''
    generate a sample accoding to text and length
    '''
    width = int(11*length)
    height = 25
   
    img = Image.new('RGB',(width,height))
    draw = ImageDraw.Draw(img)
    imgs = []

    for k in range(10):
        fontsize = int(16+random.random())
        font = ImageFont.truetype('fonts/微软雅黑.ttf', fontsize)
        w0 = 2# align left
        if length>8:
            w0=6
        elif length>15:
            w0=10
        elif length>20:
            w0=14
        h0 = (height - fontsize) // 3  # start y
        draw.text((w0, h0), text, (255,255,255), font=font)
        img = np.array(img)
        img_noise = random_noise(img,mode='gaussian')
        imgs.append(img_noise)

    for j in range(len(imgs)):
        io.imsave('sample/'+str(i)+"_"+str(j)+".jpg",imgs[j])
        h.write(str(i)+"_"+str(j)+".jpg"+'*'+text+'\n')

    return len(imgs)


print("Begin Generating Data:")
nums = 0
vocab = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k',
        'l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G',
        'H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','/','_','#',' ']
with open('items2.txt','w') as g:
    h = open('labels2.txt','w')
    for i in range(5000):
        m = random.randint(10,20)
        text = ''.join(random.sample(vocab,m))#m个char,不定长
        print(text)
        num  = gen_image(h,i,text,m)
        nums +=num
        g.write(text+'\n')
    h.close()
       
print("Generate %d images!!!!"%nums)
print("End Generating.")
