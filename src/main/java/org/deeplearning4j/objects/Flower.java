package org.deeplearning4j.objects;

import org.jol.core.MLItem;
import org.jol.core.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;

/*
1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm */

public class Flower extends MLItem {
	
  Double s_length;
  Double s_width;
  Double p_length;
  Double p_width;

  public Flower(INDArray slice, MLModel model, String[] data) {
    super(slice, model);
	s_length = Double.parseDouble(data[0]);
    s_width = Double.parseDouble(data[1]);
    p_length = Double.parseDouble(data[2]);
    p_width = Double.parseDouble(data[3]);
  }

  public String toString() {
    return "sepal length: "+s_length+" sepal width: "+s_width+" petal length: "+p_length+" petal width: "+p_width + " label: "+getLabel();
  }
}
