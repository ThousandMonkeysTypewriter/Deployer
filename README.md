# Java Objects Learning

JOL is a service that enables developers to quickly and easily load and implement Machine Learning models logic inside simple Java objects.

---
## Main Features

- the division between ML models and runtime objects helps avoid mixing of functionality
- change the model or switch frameworks while Java objects remain unchanged
- no extensive knowledge of Machine Learning necessary to use, load or train models
- allows using both ML and standard logic inside Java objects

## DL4J implementation examples
Let's suppose your goal is to create a function that will sort out flowers. Suppose you have an ML model that can distinguish between different kind of flowers based on its description.

First, we load saved and trained model using model configuration.

Then,  we create features that will be fed to the model.

We create an object of type Flower using text parameters, model and features.

Finally, we can add the Flower to the HashMap, using Flower's type as a key, and Flower object itself as a value;

Outup after bunch of flowers were created;

### Text analysis

 - Feed list of reviews (IMDB) to get sentiment analyze. Use previously trained model, or train model and then feed new reviews;
First, create the model from conf. 

`MLModel model = new MLModel(conf);`

From each review (text) we create Object, then feed it to model and label this item as positive or negative.

`    String text = FileUtils.readFileToString(files[1]);`
`    Review review = new Review(model.prepareFeatures(text), model, text);`

> /src/main/java/org/jol/ReviewsFeeder.java

- data classification (sort animals by class in CSV)

`    Animal animal = new Animal(slice, model, data.get(i));`
`    String label = animal.getLabel();`
`    animals.get(label).add(animal);` 

> /src/main/java/org/jol/AnimalsClassifier.java
> /src/main/java/org/jol/IrisClassifier.java


### Image analyzis

> dl4j-examples/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/

git update-index --assume-unchanged src/main/resources/images/model.zip
git update-index --assume-unchanged src/main/resources/review/GoogleNews-vectors-negative300.bin.gz
git update-index --assume-unchanged src/main/resources/review/dl4j_w2vSentiment/aclImdb_v1.tar.gz