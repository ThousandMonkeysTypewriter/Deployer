# Java Objects Learning

JOL is a service that enables developers to quickly and easily load and implement Machine Learning models inside simple Java objects.

---
## Main Features

**Java development friendly**:  No extensive knowledge of Machine Learning is necessary to use ML functions in runtime

**Integrity**: Objects encapsulate ML functions (prediction etc.), avoid external calls

**Mixing of techniques**: Allow using both ML and standard logic inside Java objects

**Asynchronous development**: Objects are not affected by modifications of the model

**Separation of concerns:**  Separation of ML models and objects helps to avoid mixing of the methods

## Example
<img src="https://nayname.github.io/diagram.jpg?noresize"> 

Let's suppose we have a document containing [description](https://github.com/nayname/JOL/blob/master/src/main/resources/flowers/iris.txt) of flowers and we want to turn it [into the list of objects of the type Flower](https://github.com/nayname/JOL/blob/master/src/main/java/org/deeplearning4j/IrisClassifier.java). This object will contain parameters from the document (sepal length, sepal width, etc.) and a label - which type of flower that is. The label is not provided by the document so we will use our ML model to derive the flower's label from it's parameters.

First, we load saved and trained model using the model configuration.

`MLModel model = new MLModel(conf);`

Then, we load the document. Each row represents one flower. The first row looks like this:

`5.1,3.5,1.4,0.2,0`

Using the model, model features (row) and parameters ("sepal width" etc.) we create new Flower. 

`Flower iris = new Flower(row, model, parameters.get(i));`


It will have the following fields

`sepal length: 5.1 sepal width: 3.5 petal length: 1.4 petal width: 0.2 label: Iris Setosa`

The label was predicted by the model

Finally, after analyzing all the rows, we will have the HashMap consisting of 41 Iris Virginica, 59 Iris Versicolour, 50 Iris Setosa. We can use these objects later in our program.

For more info about the library structure please check out this [JavaDoc](https://nayname.github.io/javadoc/org/jol/core/package-summary.html)

---

[Deeplearning4J implementation examples](https://github.com/nayname/JOL/blob/master/DL4J.md)

---


## Build and Run

Use [Maven](https://maven.apache.org/) to build the examples.

```
mvn clean package
```
The simplest way to  run an example is to call Java with the following inputs:

 - Path the JAR  
 - Chosen example's class as the main class

```
java -cp .:target/JOL-0.9.1-bin.jar org.deeplearning4j.IrisClassifier
```

Also, there is an option to create and train the model from scratch. 

```
java -cp .:target/JOL-0.9.1-bin.jar org.deeplearning4j.IrisClassifier create
```

Every example, except ImagesClassifier comes with the already trained model.