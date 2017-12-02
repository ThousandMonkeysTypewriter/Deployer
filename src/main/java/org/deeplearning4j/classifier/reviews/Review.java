package org.deeplearning4j.classifier.reviews;

import java.util.Map;

import org.jol.objects.MLItem;
import org.jol.objects.MLModel;
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
