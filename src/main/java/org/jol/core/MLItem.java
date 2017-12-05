package org.jol.core;

import org.deeplearning4j.objects.Animal;
import org.deeplearning4j.objects.Flower;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Parent class that can be used in objects to get some data from Machine Learning models
 * 
 * @author  Fedor Soprunov
 * @see     Animal
 */
public class MLItem {
  
  MLModel model;
  
  private INDArray features;
  private INDArray output;
  
  public MLItem(INDArray features_, MLModel model_) {
    model = model_;
    features = features_;
    
    output = model.output(features);
  }

  public String getLabel() {
    return model.getLabel(output);
  }
  
  public INDArray getOutput() {
    return output;
  }
}
