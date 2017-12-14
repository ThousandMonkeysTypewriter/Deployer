package org.examples.dl4j.objects;

import org.jol.core.MLItem;
import org.jol.core.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Image extends MLItem {
  
  private String path;
  private String animal;

  public Image(INDArray prepareFeatures, MLModel model, String text_) {
    super(prepareFeatures, model);
    path = text_;
    animal = getLabel();
  }
  
  public String toString() {
    return "path: "+path+", animal: "+animal;
  }
}
