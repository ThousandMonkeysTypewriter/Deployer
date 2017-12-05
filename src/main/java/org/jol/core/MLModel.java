package org.jol.core;

import java.io.IOException;
import java.util.Map;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Interface between Java objects and different frameworks models </br>
 * 
 * Item -> MLItem -> MLMolde -> Framework Model
 * 
 * @author  Fedor Soprunov
 */

public interface MLModel {

  void saveToDisk() throws IOException;
  
  void restore(String modelLocation) throws IOException;
  
  void fitNormalizer(DataSet input);

  INDArray prepareFeatures(String input) throws IOException, InterruptedException;
  
  INDArray prepareFeatures(DataSet test, boolean normalize) throws IOException, InterruptedException;

  Map<Integer, String> getLabels();
  
  Model getModel();
  
  String getLabel(INDArray output);

  INDArray output(INDArray features);

  void trainModel() throws Exception;
}
