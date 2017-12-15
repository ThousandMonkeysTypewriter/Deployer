package org.jol.core.items;

import org.examples.dl4j.objects.Animal;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.jol.core.MLModel;

public class MLString {
    
  private String label;
  private MLModel model;
  /**
   * Creates part of the object that generates the label
   * 
   * @param features_ input data for the model
   * @param model_ loaded Neural Network model
   */
  public MLString (MLModel model_) {
    model = model_;
  }

  /**
   * Returns string representation of the label
   */
  public String getString() {
    return label;
  }
}
