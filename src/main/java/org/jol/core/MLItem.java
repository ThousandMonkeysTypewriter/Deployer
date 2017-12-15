package org.jol.core;

import org.examples.dl4j.objects.Animal;
import org.examples.dl4j.objects.Flower;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Parent class that can be used in objects to get some data from Machine Learning models
 * 
 * @author  Fedor Soprunov
 * @see     Animal
 */
public class MLItem {
  
  private MLModel model;
  
  private INDArray features;
  private INDArray output;
  
  private String label;
  /**
   * Creates part of the object that generates the label
   * 
   * @param features_ input data for the model
   * @param model_ loaded Neural Network model
   */
  public MLItem(INDArray features_, MLModel model_) {
    model = model_;
    features = features_;
    
    output = model.output(features);
  }

  /**
   * Returns string representation of the label
   */
  public String getLabel() {
    return model.getLabel(output);
  }
  
  /**
   * Output of the model
   */
  public INDArray getOutput() {
    return output;
  }
  
  public void setFeatures(INDArray f) {
    features = f;
  }
  
  public INDArray getFeatures() {
    return features;
  }
}
