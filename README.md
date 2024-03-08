What is this?
-----------
This is a number recognition software made from scrath in cpp. 
No external libraries are used apart from opencv which is only used 
as a way to visually interact with the program.

Compliling
----------
If you wish to use the gui, before compliling be sure to include the opencv library.
```bash
make all
```

How to
------
- Binaries are stored in bin/
- What each binary does should be clear from the name
- When train is done running, data from the network in saved in data/
- Functions are provided to save, load and interact with the network

How it works?
-------------
The program uses a neural network to predict a number drawn on 
an one channel 28x28 image in which pixels range from 0 - 255.
A trained neural network takes the image as an input and outputs 
out what number it thinks it is between 0 - 9. Training the network 
is done through a process called backpropagation, which is just 
gradient descent on steroids.

Future Plans
---------------------------
Even though the program is 95% accurate with training and 92% accurate 
with testing data, this approach is not ideal. The network is 
very limited on what data can be inputted and what it can be used 
for. The network is poor at predicting numbers drawn outside of the center of 
the 28x28 grid. To amend this, we can change the progam to run an input 
image through a CNN(Convolutional Neural Network) and predict the number 
based on the structure of the input rather than the pixel values.

Resources
----------
- (All the math you need to know) https://explained.ai/matrix-calculus/
- (opencv for linux) https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
- (opencv for windows) https://opencv.org/releases/
- (MNIST database) http://yann.lecun.com/exdb/mnist/
- https://www.youtube.com/watch?v=aircAruvnKk&t=1037s

Screenshot
----------
![FIrstNeuralNet](https://user-images.githubusercontent.com/46787561/175752971-bf3f7ff9-7884-471a-8c82-2ca11367de9d.png)
