What is this?
-----------
This is a number recognition software made from scrath in cpp. 
No External libraries are used apart from opencv which is only used 
as a way to visually interact with the program

Compliling
----------
If you wish to use the gui, before compliling be sure to include the opencv library
To compile run: make all

How to run
----------
- To train simply run: bin/train
- To draw and guess numbers run: bin/gui

How it works?
-------------
The program uses a neural network to predict a number drawn on 
an one channel 28x28 image in which pixels range from 0 - 255.
A trained neural network takes the image as an input and outputs 
out what number it thinks it is between 0 - 9. Training the network 
is done through a process called backpropagation, which is just 
gradient descent on steroids.

The bad and soon to be good
---------------------------
Even though the program is 95% accurate with training and 92% accurate 
with testing data, this approach sucks balls. The neural network is 
very limited on what data can be inputted in, and what it can be used 
for. It also sucks at predicting numbers drawn outside of the center of 
the 28x28 grid. To amend this, we can change the progam to run an input 
image through a CNN(Convolutional Neural Network) and predict the number 
based on the structure of the input rather than the pixel values.

But guessing numbers is lame. Next project will implement a CNN for a 
facial recognition library 

Reasources
----------
- (All the math you need to know) https://explained.ai/matrix-calculus/
- (opencv for linux) https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
- (opencv for windows) https://opencv.org/releases/
- (MNIST database) http://yann.lecun.com/exdb/mnist/

If you dont know what a neural network even is and you dont like reading scholar write ups, 
I HIGHLY recommend 3blue1brown: https://www.youtube.com/watch?v=aircAruvnKk&t=1037s

Screenshot
----------
![FIrstNeuralNet](https://user-images.githubusercontent.com/46787561/175752971-bf3f7ff9-7884-471a-8c82-2ca11367de9d.png)
