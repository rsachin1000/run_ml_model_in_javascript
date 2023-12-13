# Run ML Model in Javascript

Run ML Model in Javascript on user's devices

## Objective:

Our objective is to take a CNN (Our example: `Image Classification Model`), convert it into formats such that it can be run in Javascript. Why would we ever want to run a ML Model in Javascript??? Answer: Suppose we want to develop an application which requires processing of `user/client` data on user's device (eg mobile or pc) itself. And often times the user would be concerned about his/her `privacy`. Then the logical way to approach this problem would be to take the user data and either process it on the user device itself or we can remove user identifiers from it.

## What are the complexities?

Machine learning models require `specialized environment` to load them into memory and run inference on input data. That environment is often provided with the help of libraries like PyTorch, TensorFlow, etc. But how would to enforce installation of external packages on users devices. Not possible, right!! So there is a way we can avoid doing all of this. We can convert our model into `ONNX` format, for which we have javascript libraries. Then this ONNX model can be loaded using javascript and run on user's device itself.

## Our Example

### Data and Prerequisities:

1. CIFAR-10 Dataset. [Source link](https://www.cs.toronto.edu/~kriz/cifar.html).
2. An Image Classification Model. We have all the model related files in this repo. However, if you want to go into the details of creating the model, training it, and making inference, visit [here](https://medium.com/bitgrit-data-science-publication/building-an-image-classification-model-with-pytorch-from-scratch-f10452073212).
