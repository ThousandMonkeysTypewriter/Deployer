## Deeplearning4J implementation examples

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



























