Pruning with pytorch.

Motivation: While Deep learning every day achieves the new state of the art results, it manages to do so with hefty costs. It uses lots of data and compute power to reach to any result that is sufficiently useful for actual day to-day applications. It is not possible to have such hardware with high computational power at the user’s end, imagine having a mobile of size multiple laptops, why even call it mobile? 😊 Computational and energy efficiency is the major problem in current deep learning systems. We need to be able to do same tasks with less computationally extensive systems without major loss in accuracy. Pruning in deep learning is one of the methods that helps just to do that.

NN pruning: It is the process with which we aim to reduce the size and complexity of the NN by reducing the computations that NN does at the time of inference. Not every neuron or every connection in the deep NN helps with same extent to optimize the cost function. Redundant parts of the NN are removed without having significant impact on the accuracy at the benefit of a lot of less space and computations. It does so by ranking each neuron for the extent to which that neuron helps in optimizing the cost function and removing least important NN / connection. 

There are mainly two methods,
1.	Wight pruning: Setting a weight to zero in the weight matrix effectively removes a connection between two neurons.
2.	Neuron pruning: Setting a complete column in the weight matrix effectively removes all the connections to a neuron disabling it from taking part in any computation.

Both methods result in what is called as sparse network. Sparse network learning is found to be further improved with hardware acceleration [1].
It is recommended that pruning should be done iteratively, as with high pruning in a single iteration NN might not recover/heal from loss of weights and/or neurons. Pruning followed by a retraining is one iteration, after many such iterations the minimum number connections could be found. [2]
While Neural Network pruning is a universal concept, I found it mainly used in convolution neural network. May be simply because CNNs typically are very deep and when trained for vision tasks do not suffer a lot from pruning as opposed to say LSTM or RNN. Because CNN for vision has features we are interested in spatial dimension unlike RNN/LSTM which have it in temporal time dimension.

There are also different ways in which pruning is applied – 
1.	Unstructured: It is applied generally to the weights
2.	Structured: It is applied to filters or layers which are larger parts of in the NN
3.	Global: It is applied across complete NN for selected type of the part of the NN such as weights/filters or layers.

Recently there have been research around training the module completely from the scratch after Pruning, which seems promising.

I have created the CIFAR CNN to see the effects of the pruning. I have used the built in APIs of the pruning in Pytorch with some custom pruning functions as well. I have used the unstructured L1 pruning from the pytorch, but I would expect even more good results with structured and global pruning. Please Refer to the attached ipynb.

References:
[1] https://arxiv.org/pdf/1812.01164.pdf    
[2] https://papers.nips.cc/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf
