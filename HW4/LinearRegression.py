#Muhammad Rizwan Malik
#Hamza Belal KAzi


###<<<<<<<<, LINEAR REGRESSION ALGORITHM >>>>>>>>>>>>>>>>>>>>>
import numpy as np
from numpy import loadtxt
from numpy.linalg import inv

### No Iterations just one shot optimization in this using data array##

#Here X is both first 2 coordinate and prediction is  3rd coordinate

np_data_points=0
with open('linear-regression.txt', 'r') as in_file:
    np_data_points = loadtxt('linear-regression.txt', delimiter=',')


y=np_data_points[:,2]
X0_Col=np.ones(   ( len(np_data_points),1 )         )
D = np.append( X0_Col,  np.delete(np_data_points, 2, axis=1) , 1)
# append a column of 1's to D as X0

D= np.transpose(D)
D_T= np.transpose(D)

DD_T= np.matmul(D,D_T)

DD_T_Inverse= inv(DD_T)


D_cross_y= np.matmul(D,y)

W= np.matmul( DD_T_Inverse, D_cross_y)

print("Weight vector  W = [" + str(W[0]) +", " + str(W[1]) +", " + str(W[2])   +"]")

