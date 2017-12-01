# Java Objects Learning

JOL is a service that enables developers to quickly and easily load and implement Machine Learning models logic inside simple Java objects.

---
## Main Features

- division between ML models and runtime objects, help avoid mixing of functionality
- change the model or switch frameworks while Java objects it remain unchanged
- allows mixing of ML and standart logic inside Java objects

## Example
Lets suppose your goal is to create a function that will sort out flowers. Suppose you have an ML model that can distinguish between different kind of flowers based on its description.

First, we load saved and trained model using model configuration.

Then,  we create features that will be fed to the model.

We create an object of type Flower using text parameters, model and features.

Finally, we can add the Flower to the HashMap, using Flower's type as a key, and Flower object itself as a value;

Outup after bunch of flowers were created;

## Text analysis

 - Feed list of reviews (IMDB) to get sentiment analyze. Use previously trained model, or train model and then feed new reviews;
First, create the model from conf. 

`MLModel model = new MLModel(conf);`

From each review (text) we create Object, then feed it to model and label this item as positive or negative.

`MLItem review = new MLItem(text, model);`

> /dl4j-examples/jol/src/main/java/org/deeplearning4j/sentiment/reviews

- suggest
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/basic/BasicRNNExample.java

- data classification (sort animals by class in CSV)
> /dl4j-examples/jol/src/main/java/org/deeplearning4j/animals/reviews

- iris 
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExample.java

> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/


- draw plot(??)
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVPlotter.java
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/feedforward/classification/PlotUtil.java

## Image analyzis

- image classification
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/


## Log analyzis

- UI 
> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/userInterface/

- log anomaly detection

- ranker

- dating site (text classification + local ranker)