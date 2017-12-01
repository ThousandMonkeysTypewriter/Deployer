package org.deeplearning4j.classifier.animals;

import org.jol.objects.MLItem;
import org.jol.objects.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Animal extends MLItem {
  
  Double yearsLived;
  String eats;
  String sounds;
  Double weight;

  public Animal(INDArray slice, MLModel model, String[] data) {
    super(slice, model);
    yearsLived = Double.parseDouble(data[0]);
    eats = data[1];
    sounds = data[2];
    weight = Double.parseDouble(data[3]);
  }

  public String toString() {
    return "sounds: "+sounds+", eats: "+eats+", weight: "+weight+" lived: "+yearsLived;
  }
}
