package org.jol.core;

import java.io.IOException;
import java.util.Map;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public interface MLModel {

  void saveToDisk() throws IOException;
  
  void restore(String modelLocation) throws IOException;
  
  void fitNormalizer(DataSet input);

//  void normalize (INDArray slice);

  INDArray prepareFeatures(String input) throws IOException, InterruptedException;
  
  INDArray prepareFeatures(DataSet test, boolean normalize) throws IOException, InterruptedException;

  Map<Integer, String> getLabels();
  
  Model getModel();
  
  String getLabel(INDArray output);

  INDArray output(INDArray features);

//  String eval() throws IOException;
  
  void trainModel() throws Exception;
}
