#Muhammad Rizwan Malik
#Hamza Belal KAzi

###<<<<<<<<, POCKET ALGORITHM >>>>>>>>>>>>>>>>>>>>>

import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
# below code for checking random point or accuracy
def get_num_errors():
    error_counter=0
    for idx in range(0,2000):
        final_test= W0*1 + W1*input[idx][0]  +W2*input[idx][1] + W3*input[idx][2]

        if  final_test <= 0 and input[idx,3] == 1:  #error case one keep count
            error_counter=error_counter + 1
        if  final_test >= 0 and input[idx,3] == -1: 
            error_counter=error_counter + 1

    return error_counter

 #3D problem so 3 wt variables + threshold(W0)
W1=5
W2=5
W3=5 
W0=4  # -threshold woith X0=1

step_size=0.01#0.008

np_data_points=0
with open('classification.txt', 'r') as in_file:
    np_data_points = loadtxt('classification.txt', delimiter=',')


input =   np.delete(np_data_points, 3, axis=1) # using last column as labels  and delete older labels

iter=0
error_size=7000
num_misclassify=[0]*error_size
while (True):       
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
                num_misclassify[iter]= get_num_errors() # call function to calculate and store number of misclassified pts
                iter=iter+1

            elif value >=0 and point[3]==-1: # Case2
                W0=W0 - step_size* 1
                W1=W1 - step_size*point[0]
                W2=W2 - step_size*point[1]
                W3=W3 - step_size*point[2]
                num_misclassify[iter]= get_num_errors() # call function to calculate and store number of misclassified pts
                iter=iter+1

            if iter>=error_size :  # run only 7000 iterations
                break
        if iter>=error_size :  # run only 7000 iterations
             break     # break from both loops

print("Weight vector  W = [" + str(W0) +", " + str(W1) +", " + str(W2) +", "+ str(W3)  +"]")


downSample=100
y=[0]* int (error_size / downSample)
y_idx=0
for i in range(0,len(num_misclassify)): # down sampling the errors array fro better display
    if i % downSample ==0: #down sample by 100
        y[y_idx]=num_misclassify[i]
        y_idx= y_idx+1




fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 7000, len(y) )
ax.plot(x, y)
plt.show()

numWrong= num_misclassify[-2]   #get_num_errors()
Accuracy= 100 - (  (numWrong/2000) * 100   )
print("Accuracy =  " + str(Accuracy) + "%")

      

 
