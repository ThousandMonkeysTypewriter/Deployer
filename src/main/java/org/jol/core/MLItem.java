package org.jol.core;

import java.util.HashMap;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MLItem {
  
  MLModel model;
  
  private INDArray features;
  private INDArray output;
  
  private HashMap<Integer, Float> params = new HashMap<>();
  
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
  
//  public String eval() throws IOException {
//    return model.eval();
//  }
}
