# Pytorch Custom One Layer Model
This project is a quite simple and easy-to-understand implementation of a one-layer neural network using PyTorch. I implemented `forward` and `backward` functions of both `MyLinearLayer` and `CrossEntropyLoss` class from scratch to gain a deeper understanding of PyTorch and its autograd functionality. This PyTorch project is important for me as it provided essential insights that form the basis of my thesis.

## Project Overview
This code consists of two key classes: `MyLinearLayer` and `CrossEntropyLoss`. They are inherited from `Module.nn` in PyTorch mainly used for neural network usages.
PyTorch users are able to build and run models easily since autograd helps them calculate backward and forward pass automatically.  I was surely aware of this helpful feature of PyTorch, but I was willing to actually comprehend its functionality. Therefore, I wrote backward and forward pass from scratch.
Also, I used MNIST dataset as it is simple and understandable.

Here's a brief explanation of my classes:

### MyLinearLayer
In the constructor, I initialized weights and bias of my model with output and input size. They have to be assigned as Parameters to be optimized. Then, in forward function I called the LinearFunction class and passed input, weights and bias.

### LinearFunction
This class which inherits from `torch.autograd`, actually does all the forward and backward pass of our linear layer. 
In `forward` function, I simply calculated:  $$ x.w^T+b $$

In `backward` function, I had to calculate the derivative of output w.r.t to each parameter. And by multiplying these derivatives and grad_output, the derivatives of each parameter w.r.t Loss is achieved.
$$ \frac{dLoss}{dw} =  \frac{dLoss}{dy} \frac{dy}{dw}$$ 
$$ \frac{dLoss}{dx} = \frac{dLoss}{dy}\frac{dy}{dx} $$

### CrossEntropyLossFunction
**forward:**
After receiving output and target from `CrossEntropyLoss` class, In forward method, I calculated softmax and logarithm of outputs, then built a one-hot-label vector to represent the true and false classes. 
- For example, assume that the true label is '8', the one hot label vector would be vector with size of 10, which all elements are zero, except the eighth element which is 1.

After that, we calculate loss based on this formula:
$$ Loss = - \sum \tau . y$$
tau is one_hot_labels and y is the output after softmax.

**backward:**
In backward , I had to again calculate the derivative of Loss w.r.t y, which is:
$$ \frac{dLoss}{dy}$$
now, it makes sense! because after calculating the above derivative, the backward function of CrossEntropyLoss actually sends this to backward function of LinearLayerFunction as `grad_output`. Then `grad_output` is multiplied with each gradient of x and w and it builds `grad_weight` and `grad_input`. These gradients are further used for optimizing and updating. You now see how beautiful and organized it works!

The other classes are pretty straight forward :) They all join together and build a simple neural network.

## Results
I set the epoch number to 50, learning rate to 0.01 and momentum to 0.9. Here are the training and test results:


## Usage
Simply run `oneLayerModel.ipynb` in Google Colab or Jupyter :)

## Contact
Created by [mahya.ehsanimehr@gmail.com](mailto:mahya.ehsanimehr@gmail.com) - feel free to contact me!