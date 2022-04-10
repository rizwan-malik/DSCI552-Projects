#Muhammad Rizwan Malik
#Hamza Belal KAzi

###<<<<<<<<, LOGISTIC REGRESSION ALGORITHM >>>>>>>>>>>>>>>>>>>>>

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import math
# below code for checking random point or accuracy

threshold=0.5
def get_num_errors():
    error_counter=0
    debug_return=0
    for idx in range(0,len(input)):
        final_prob= W[0]*1 + W[1]*input[idx][0]  +W[2]*input[idx][1] + W[3]*input[idx][2]

        final_prob= sigmoid(final_prob)

        if  final_prob <= threshold and input[idx,3] == 1:  #error case one keep count
            error_counter=error_counter + 1
        if  final_prob >= threshold and input[idx,3] == -1: 
            error_counter=error_counter + 1
        if idx==1:
            debug_return=final_prob## delete this debog
    return   error_counter

def cost():
    sum=0
    for  row in input:
        Xi=[1,row[0],row[1],row[2]]
        yi=row[3]
        pow=-yi* np.dot(W,Xi)
        takeLog= 1 + math.exp(pow)
        sum=sum+math.log(takeLog)
    return sum/N


def sigmoid(s):
   return math.exp(s)/(1+ math.exp(s)  )

 #3D problem so 3 wt variables + threshold(W0)
W=[0.1,0.2,0.3, -0.4]
#W=[1,2,3,4]
N=2000

neeta= 0.11#0.01  # learning rate neeta

np_data_points=0
with open('classification.txt', 'r') as in_file:
    np_data_points = loadtxt('classification.txt', delimiter=',')


input =   np.delete(np_data_points, 3, axis=1) # using last column as labels  and delete older labels

iter=0
error_size=7000
num_misclassify=[0]*error_size
while (True):       

    numWrong= get_num_errors()
    Accuracy= 100 - (  (numWrong/2000) * 100   )
    print("Accuracy =  " + str( Accuracy ) + "%")


    sum=0
    batch =0
    for row in input:
       # batch=batch+1
        yi=row[3]
        Xi=[1] # size must be 1x4 final
        Xi.append(row[0] )
        Xi.append(row[1] )
        Xi.append(row[2] )

        power= yi * np.dot(W,Xi)
        denom= 1 + math.exp(power)

        sum = sum + (yi/denom)*np.array(Xi)
        # quicker smaller chunk update trial
        # if batch % 10==0 : # batch size of 10
        #     gradient_error= sum/100
        #     W = W - neeta * gradient_error 

    gradient_error= sum/N
    W = W + neeta * gradient_error 



    iter=iter+1
    
    if iter>=error_size :  # run only 7000 iterations
            break     # break from both loops

print("Weight vector  W = [" + str(W[0]) +", " + str(W[1]) +", " + str(W[2]) +", "+ str(W[3])  +"]")



numWrong= get_num_errors()
Accuracy= 100 - (  (numWrong/2000) * 100   )
print("Final Accuracy =  " + str(Accuracy ) + "%")

      

 
