import numpy as np
import cv2
import math

import sys
import os

# Read the list of training files. Splitlines method is very useful. Will try to use it for reading the pgm file as well
f = open("downgesture_train.list")
train_list = f.read().splitlines()
f.close()

f = open("downgesture_test.list")
test_list = f.read().splitlines()
f.close()


f = 'gestures\gestures\A\A_down_1.pgm' 


def read_pgm(pgmf, file_list=None):
	img = cv2.imread(pgmf,cv2.IMREAD_GRAYSCALE)
	img=np.array(img)
	input= np.reshape(img, ( 960,1))
	return input

def sigmoid ( num ):
	return 1/(1+ np.exp(-num) )

def square_error(p, actual):
   
   
   return  math.pow(actual-p,2) 




##########################################################################

AllTrainImages=[]
AllLabels=[]

for file_name in train_list:
	img=read_pgm(file_name)
	AllTrainImages.append(img)
	if 'down' in file_name:
		AllLabels.append(1)
	else:
		AllLabels.append(0)


AllTestImages=[]
AllTestLabels=[]

for file_name in test_list:
	img=read_pgm(file_name)
	AllTestImages.append(img)
	if 'down' in file_name:
		AllTestLabels.append(1)
	else:
		AllTestLabels.append(0)




Alpha= 0.1 # learning rate

Vsigmoid=np.vectorize(sigmoid) # note numpy also has its own exp which can be applied in sigmoid in a vector way
Vsquare_error= np.vectorize(square_error)


batch_size=1
L2_neurons=  100
W_12= np.matrix(np.random.rand (L2_neurons,960) -0.5 )
W_23= np.matrix(np.random.rand(1 ,L2_neurons)   -0.5 )

totalBatches= 184 #math.floor(len(AllTrainImages) /batch_size ) 
totalEpochs= 1000 #math.ceil(180*1000 /len(AllTrainImages))#18

for epoch in range(0,totalEpochs):
	CorrectPredictions=0
	for batchesLoop in range( 0, totalBatches):
	   # print('Batch number '+str(batchesLoop)+" out of " + str(totalBatches) )
		BigDelta_1=0 # innitialize these properly with zero matrix of right size
		BigDelta_2=0

		#image_data= np.genfromtxt(trainingData,delimiter=',',max_rows= batch_size, dtype=float, skip_header=batch_size*batchesLoop)
		image_data=np.reshape(AllTrainImages[batchesLoop* batch_size :(batchesLoop+1)* batch_size] ,(960,1) )
		label=  AllLabels[batchesLoop* batch_size :(batchesLoop+1)* batch_size]#np.genfromtxt(trainingLabel,delimiter=',',max_rows= batch_size, dtype=float, skip_header=batch_size*batchesLoop)
		#image_data=np.array(image_data)# image_data=np.matrix(image_data)


		
		expected_output = label
		###>>> Feed Forward propagation<<<###########
		Ave_pixel=0#average of pixel values
		output_L1= (image_data - (Ave_pixel) )/255.0
		input_L2=W_12* output_L1
		output_L2= Vsigmoid(input_L2)


		input_L3=W_23*output_L2
		output_L3= Vsigmoid(input_L3)


		cost= np.sum( Vsquare_error(output_L3,expected_output)  )

		###-->>>>> back propagation computations (NOTE:BigDelta means a the partial derivative w.r.t weights of error.. small delta is used for its computation)  <<---##3
		# compute delta for final layer  
		SmallDelta_3=np.subtract(output_L3, expected_output) # delta of final layer
		# compute delta for all previous layers except 1 by back  propaation formula
		SmallDelta_2=  np.multiply(W_23.transpose()* SmallDelta_3  ,np.multiply( output_L2 , (1-output_L2) )  )


		BigDelta_2=BigDelta_2 + SmallDelta_3 *output_L2.transpose()
		BigDelta_1= BigDelta_1 + SmallDelta_2 * output_L1.transpose()




		###  OUT SIDE FOR LOOP of BATCH   NOTE: may have to make 2 cases for when Bias term and other terms  Dij   j=0  j!=0 ( add regularization for non bias terms)
		D_1= (1/batch_size)  * BigDelta_1
		D_2= (1/batch_size)  * BigDelta_2


		# Weights update
		W_23= W_23  - Alpha * BigDelta_2
		W_12= W_12  - Alpha * BigDelta_1

		# print(label)
		# print(batchesLoop)
		# print( W_23 )

		#print(" ")


	#end of epoch loop
	# evaluate on test data set and also write its output
	PBigDelta_1=0 # innitialize these properly with zero matrix of right size
	PBigDelta_2=0

	# Pimage_data= np.genfromtxt(testData,delimiter=',' , dtype=float, skip_header=0)
	# Pimage_data= np.matrix(Pimage_data)

	Pimage_data=  np.reshape(AllTestImages,(83,960))  #np.matrix(AllTestImages)
	 ###>>> Feed Forward propagation<<<###########
	Ave_pixel=0 #36.5#average of pixel values
	output_L1=   (Pimage_data.transpose() - (Ave_pixel) )/255.0
	input_L2=W_12* output_L1
	output_L2= Vsigmoid(input_L2)

	input_L3=W_23*output_L2
	output_L3= Vsigmoid(input_L3)
	
	predictedNums=  np.where(output_L3<=0.5,0,1)  # output_L3#np.argmax(output_L3,axis=0)

	# # Save Numpy array to csv
	# np.savetxt('HamzaTest.csv', predictedNums, delimiter='\n', fmt='%d')
	# Test_label=np.genfromtxt('test_label.csv',delimiter=',', dtype=float)
	Test_label=AllTestLabels
	CorrectPredictions=np.sum(predictedNums==Test_label)


	print(" Epoch number :" + str(epoch))
	print( predictedNums)
	print("Squared Error Loss: "+str(cost) )
	print("Accuracy on whole after epoch:"+ str(CorrectPredictions*100/( len(Test_label) )  )   )



# # evaluate on test data set and also write its output
# PBigDelta_1=0 # innitialize these properly with zero matrix of right size
# PBigDelta_2=0

# Pimage_data= np.genfromtxt(testData,delimiter=',' , dtype=float, skip_header=0)
# Pimage_data= np.matrix(Pimage_data)
#  ###>>> Feed Forward propagation<<<###########
# Ave_pixel=36.5#average of pixel values
# output_L1= 0.25*  (Pimage_data.transpose() - (Ave_pixel) )/255.0
# input_L2=W_12* output_L1
# output_L2= Vsigmoid(input_L2)

# input_L3=W_23*output_L2
# output_L3= Vsigmoid(input_L3)
# predictedNums=np.argmax(output_L3,axis=0)
# # Save Numpy array to csv
# np.savetxt('test_predictions.csv', predictedNums, delimiter='\n', fmt='%d')

