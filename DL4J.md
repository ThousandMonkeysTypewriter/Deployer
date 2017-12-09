## Deeplearning4J classifier examples

### Sentiment analysis

Main class

- org.deeplearning4j.ReviewsClassifier

Gets a sentiment from the tex sample (IMDB review). 


From each review we create Object, then feed it to the model and label this item as positive or negative.

`    String text = FileUtils.readFileToString(files[1]);`
`    Review review = new Review(model.prepareFeatures(text), model, text);`

### Text classifier

Main class

- org.deeplearning4j.AnimalsClassifier

sort animals descriptions by types

`    Animal animal = new Animal(slice, model, data.get(i));`
`    String label = animal.getLabel();`
`    animals.get(label).add(animal);` 

### Image classifier

Main class

- org.deeplearning4j.AnimalsClassifier

sort animals descriptions by types

`    Animal animal = new Animal(slice, model, data.get(i));`
`    String label = animal.getLabel();`
`    animals.get(label).add(animal);` 

### Iris classifier

Main class

- org.deeplearning4j.AnimalsClassifier

sort animals descriptions by types

`    Animal animal = new Animal(slice, model, data.get(i));`
`    String label = animal.getLabel();`
`    animals.get(label).add(animal);` 



























