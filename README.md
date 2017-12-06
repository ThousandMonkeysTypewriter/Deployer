# Java Objects Learning

JOL is a service that enables developers to quickly and easily load and implement Machine Learning models logic inside simple Java objects.

---
## Main Features

**Separation of concerns:**  Separation of ML models and runtime objects helps to avoid mixing of the logic

**Asynchronous development**: Modification of the model (including switching ML frameworks) or the object does not affect one another

**Java development friendly**:  no extensive knowledge of Machine Learning or Python is necessary to use, load or train models

**Mixing of techniques**: allows using both ML and standard logic inside Java objects


## Example
Let's suppose we have a document containing [description](https://github.com/nayname/JOL/blob/master/src/main/resources/flowers/iris.txt) of flowers and we want to turn it [into the list of objects of the type Flower](https://github.com/nayname/JOL/blob/master/src/main/java/org/deeplearning4j/IrisClassifier.java). This object will contain parameters from the document (sepal length, sepal width, etc.) and a label - which type of flower that is. The label is not provided by the document so we will use our ML model to derive the flower's label from it's parameters.

First, we load saved and trained model using model configuration.

`MLModel model = new MLModel(conf);`

Then, we load the document. Each row of the represents one flower. The first row looks like this:

`5.1,3.5,1.4,0.2,0`

Using the model, model features and parameters we create new Flower

`Flower iris = new Flower(slice, model, data.get(i));`


It will have the following fields

`sepal length: 5.1 sepal width: 3.5 petal length: 1.4 petal width: 0.2 label: Iris Setosa`

The label was predicted by the model

Finally, after analyzing all the rows, we will have the HashMap consisting of 41 Iris Virginica, 59 Iris Versicolour, 50 Iris Setosa. We can use these objects later in our program.

For more info about the library structure please checkout out this [JavaDoc](https://nayname.github.io/javadoc/org/jol/core/package-summary.html)

---


[Deeplearning4J implementation examples](https://github.com/nayname/JOL/blob/master/DL4J.md)