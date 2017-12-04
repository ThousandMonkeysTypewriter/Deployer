# Java Objects Learning

JOL is a service that enables developers to quickly and easily load and implement Machine Learning models logic inside simple Java objects.

---
## Main Features

**Separation of concerns:**  Separation of ML models and runtime objects helps to avoid mixing of the logic

**Asynchronous development**: Modification of the model (including switching ML frameworks) or the object does not affect one another

**Java development friendly**:  no extensive knowledge of Machine Learning or Python is necessary to use, load or train models

**Mixing of techniques**: allows using both ML and standard logic inside Java objects


## Example
Let's suppose your goal is to create a function that will sort out flowers. Suppose you have an ML model that can distinguish between different kind of flowers based on its description.

First, we load saved and trained model using model configuration.

Then,  we create features that will be fed to the model.

We create an object of type Flower using text parameters, model and features.

Finally, we can add the Flower to the HashMap, using Flower's type as a key, and Flower object itself as a value;

Outup after bunch of flowers were created;

---


[Deeplearning4J implementation examples](https://github.com/nayname/DL4J.md)

[Tensorflow implementation examples](https://github.com/nayname/JOL/TENSORFLOW.md)