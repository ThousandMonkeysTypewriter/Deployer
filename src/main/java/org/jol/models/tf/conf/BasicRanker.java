package org.jol.models.tf.conf;

import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.nn.api.Model;
import org.jol.core.MLConf;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.tensorflow.Tensor;

public class BasicRanker extends MLConf {

  private float[][] vector;

  public Model train(MLConf global_conf) throws Exception {
    // TODO Auto-generated method stub
    return null;
  }

  public float getIndex(INDArray output) {
    return vector[1][0];
  }

  public Tensor prepareFeatures(ArrayList<double[]> inputDoubles){
    float[][] matrix = new float[inputDoubles.size()][numInputs];

    for (int j=0;j<inputDoubles.size();j++) 
      for(int i=0;i<batchSizeTest;i++)
        matrix[j][i]=(float)inputDoubles.get(j)[i];
//      items.get(j).tf_matrix = matrix[j];
 
    return Tensor.create(matrix);
  }
  
    public INDArray prepareFeatures(String input) throws IOException,
      InterruptedException {
		 return null;
	}
}
