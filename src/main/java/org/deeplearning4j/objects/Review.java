package org.deeplearning4j.objects;

import org.jol.core.MLItem;
import org.jol.core.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Review extends MLItem {
  
  private String text;
  private String sentiment;

  public Review(INDArray prepareFeatures, MLModel model, String text_) {
    super(prepareFeatures, model);
    text = text_;
    sentiment = getLabel();
  }
  
  public String toString() {
    return "text: "+text+", sentiment: "+sentiment;
  }
}
