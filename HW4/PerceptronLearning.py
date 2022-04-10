#Muhammad Rizwan Malik
#Hamza Belal KAzi

###<<<<<<<<PERCEPTRON LEARNING>>>>>>>>>>>>>>>>

import numpy as np
from numpy import loadtxt
 #3D problem so 3 wt variables + threshold(W0)
W1=5
W2=5
W3=5 
W0=4  # -threshold woith X0=1

step_size=0.1

np_data_points=0
with open('classification.txt', 'r') as in_file:
    np_data_points = loadtxt('classification.txt', delimiter=',')


input=np_data_points[:,:4]

iter=0
while (True):
        iter=iter+1
        W0_old=W0
        W1_old=W1
        W2_old=W2
        W3_old=W3
        for point in input:
            value= W0*1 + W1*point[0]  +W2*point[1] + W3*point[2]
            if  value < 0 and point[3] == 1:  # CAse 1 w= w+ alpha*X
                W0=W0 + step_size* 1
                W1=W1 + step_size*point[0]
                W2=W2 + step_size*point[1]
                W3=W3 + step_size*point[2]
            elif value >=0 and point[3]==-1: # Case2
                W0=W0 - step_size* 1
                W1=W1 - step_size*point[0]
                W2=W2 - step_size*point[1]
                W3=W3 - step_size*point[2]
        if W0== W0_old and  W1== W1_old and  W2== W2_old and  W3== W3_old:  # i.e no change in W after iteration
            break

print("Weight vector  W = [" + str(W0) +", " + str(W1) +", " + str(W2) +", "+ str(W3)  +"]")


# below code for checking all point or accuracy
error_counter=0
for idx in range(0,2000):
    final_test= W0*1 + W1*input[idx][0]  +W2*input[idx][1] + W3*input[idx][2]

    if  final_test <= 0 and input[idx,3] == 1:  #error case one keep count
        error_counter=error_counter + 1
    if  final_test >= 0 and input[idx,3] == -1: 
        error_counter=error_counter + 1


Accuracy= 100 - (  (error_counter/2000) * 100   )
print("Accuracy =  " + str(Accuracy) + "%")
# if  final_test > 0 and input[idx,3] == 1: 
#     print("Both results positve for point index : " + str(idx))
# if  final_test < 0 and input[idx,3] == -1: 
#     print("Both results negative for point index : " + str(idx))

      

 
