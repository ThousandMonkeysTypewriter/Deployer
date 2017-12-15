package org.jol.models.tf.conf;

import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.nn.api.Model;
import org.jol.core.MLConf;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.tensorflow.Tensor;

public class BasicRanker extends MLConf {

  public Model train(MLConf global_conf) throws Exception {
    // TODO Auto-generated method stub
    return null;
  }

  public float getIndex(INDArray output) {
    return 0f;
  }

  public INDArray prepareFeatures(String input) throws IOException,
  InterruptedException {
    return null;
  }
}
