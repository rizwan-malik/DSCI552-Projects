
import numpy as np
from numpy import loadtxt
import math
import matplotlib.pyplot as plt

totalWords=10
maxLVL=1 # 2 dimentions  i.e 0 and 1

def readFile():
	with open('fastmap-data.txt', 'r') as in_file:
		pair_Wise_Dist = loadtxt('fastmap-data.txt', delimiter='\t',dtype=int)
	return pair_Wise_Dist

def CaclFurtherstDist(distArray,lvl=0):
    #pick random start point (e.g 1st pt always)
    
    curr_selected=6
    prev_selected= curr_selected
    storeFurthestPt=None
    # create set and keep adding selected pts to that set to check everytime

    # when 2 pts have each other as the furthest then they are globally the furthest pts
    while(1):
        max_dist=-1
        for p2 in range(0,10):
            if max_dist< getDist2Pts(curr_selected,p2,distArray,lvl):

                max_dist=getDist2Pts(curr_selected,p2,distArray,lvl)  
                storeFurthestPt=p2
        
        if storeFurthestPt== prev_selected :
            return [prev_selected,curr_selected] #  return pair of furthest selected pts

        prev_selected=curr_selected
        curr_selected= storeFurthestPt
        

       


def getDist2Pts( p1, p2,distArray,iteration=0):
    p1=p1+1 # adding 1 so that it matches text file index   as python is zero based
    p2=p2+1
    if p1==p2:
        return 0
    if p1>p2: # swap so that p1 is always bigger index point(word)
        temp=p1
        p1=p2
        p2=temp

    additionalTerm=0
    for l in range(iteration + 1):
        additionalTerm= additionalTerm + (FinalCoords[p1-1][l] -FinalCoords[p2-1][l] )**2

    for i in range(0,len(distArray) )   :
        if distArray[i][0]==p1  and distArray[i][1]==p2 :
            #construct generic distance for all iterations
            d_Original=distArray[i][2]  
            d_new= math.sqrt ( d_Original **2 - additionalTerm )
            return d_new
    
def setCoords( FinalCoords,dist,lvl=0):
    # lvl=0
    furthestPair= CaclFurtherstDist(dist,lvl)  
    a=furthestPair[0] 
    b=furthestPair[1]


    # FinalCoords[a] [lvl + 1 ]= 0
    # FinalCoords[b ] [lvl + 1 ]= getDist2Pts( a, b , dist,lvl)
    #set one coordinate based on furthest pair now and its cosine triangle formula for others  
    for i in range(0,10):
        dai_sq= getDist2Pts( a, i , dist,lvl) **2
        dab_sq=  getDist2Pts( a, b , dist,lvl) **2
        dib_sq=  getDist2Pts( i, b , dist,lvl) **2
        dab   = getDist2Pts( a, b , dist,lvl) 
        xi = ( dai_sq + dab_sq + - dib_sq ) /( 2* dab )

        FinalCoords[ i] [lvl +1]=  round(xi,2)

    
    if lvl==maxLVL:
        return FinalCoords   # base case kind of all coordinated set
    else:
        return setCoords(FinalCoords,dist, lvl+1)
    






dist=readFile()
#furthestPair= CaclFurtherstDist(dist)     
#assign 1st coordinate of furthest pt in a new array
words=0
with open('fastmap-wordlist.txt', 'r') as in_file:
	words = loadtxt('fastmap-wordlist.txt',dtype= str)
	

FinalCoords=np.zeros([10,3])

#print(FinalCoords)
FinalCoords=setCoords( FinalCoords,dist)

#print(FinalCoords)
#print(CaclFurtherstDist(dist))

fig = plt.figure()
ax = fig.add_subplot(111)
xs = FinalCoords[:, 1]
ys = FinalCoords[:, 2]

ax.scatter(xs, ys)

for i in range(0,10):
    ax.annotate(words[i], (xs[i], ys[i]))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()
