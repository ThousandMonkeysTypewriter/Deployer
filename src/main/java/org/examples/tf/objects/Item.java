package org.examples.tf.objects;

import java.util.ArrayList;

import org.jol.core.MLItem;
import org.jol.core.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;

public class Item extends MLItem {
  
  ArrayList<Row> rows;
  double score;

  public Item(INDArray slice, MLModel model, String[] data) {
    super(slice, model);
  }

  public String toString() {
    return "rows: "+rows+", score: "+score;
  }
}

class Row {
  
}