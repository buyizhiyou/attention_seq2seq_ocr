#-*-coding:utf8-*-

__author="buyizhiyou"
__date="2018-11-5"


import numpy as np 
import Levenshtein

def edit(s1,s2):
    m = len(s1)
    n = len(s2)
    D = np.zeros((m+1,n+1))
    
    for i in range(m+1):
        for j in range(n+1):
            if j==0:
                D[i][j]=i
                continue
            if i==0:
                D[i][j]=j
                continue
            if s1[i-1]==s2[j-1]:
                tmp=0
            else:
                tmp=1
            D[i,j] = min(D[i-1,j]+1,D[i,j-1]+1,D[i-1,j-1]+tmp)

    return D[m][n]

if __name__ =="__main__"  :
    s1="abcdfdafdsafdsa"
    s2="adcbb"
    print(edit(s1,s2))
    print(Levenshtein.distance(s1,s2))
    assert edit(s1,s2)==Levenshtein.distance(s1,s2),"wrong"
    '''
    In [21]: %timeit edit(s1,s2)                                                                                                        
    2.33 ms ± 89.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    In [22]: %timeit distance(s1,s2)                                                                                                    
    2.72 µs ± 64.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    '''