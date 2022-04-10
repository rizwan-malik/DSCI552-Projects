## Homework 5 - Feed Forward Neural Networks
Implement the Back Propagation algorithm for Feed Forward Neural Networks to learn the down gestures from training instances available in **downgesture_train.list**.
- The label of an image is 1 if the word **down** is in its file name; otherwise the label is 0.
- The pixels of an image use the gray scale ranging from 0 to 1.
- The image file format is **[pgm](http://netpbm.sourceforge.net/doc/pgm.html)**. Please follow the link for the format details.
- In the network, use one input layer, one hidden layer of size 100, and one output perceptron.
- Use the value 0.1 for the learning rate. For each perceptron, use the sigmoid function Ɵ(s) = 1/(1+e-s).
- Use 1000 training epochs; initialize all the weights randomly between -0.01 and 0.01.
- For the error function, use the standard squared error.
#### Deliverables:
Use the trained network to predict the labels for the gestures in the test images available in **downgesture_test.list**. Output the **predictions** and **accuracy**.
