package org.jol.core;

import java.io.IOException;
import java.util.ArrayList;
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

  /**
   * Save model to the path taken from the configuration
   * 
   * @throws IOException
   */
  void saveToDisk() throws IOException;
  
  /**
   * Activate instance from the file 
   * 
   * @param modelLocation
   * @throws IOException
   */
  void restore(String modelLocation) throws IOException;
  
  void fitNormalizer(DataSet input);

  /**
   * Featurize from the input
   * 
   * @param input
   * @return
   * @throws IOException
   * @throws InterruptedException
   */
  INDArray prepareFeatures(String input) throws IOException, InterruptedException;
  
  /**
   * Featurize from the input
   * 
   * @param test    input Dataset
   * @param normalize
   * @return
   * @throws IOException
   * @throws InterruptedException
   */
  INDArray prepareFeatures(DataSet test, boolean normalize) throws IOException, InterruptedException;
  
  void prepareFeatures(ArrayList<MLItem> items);

  Map<Integer, String> getLabels();
  
  Model getModel();
  
  /**
   * takes MLItems's input
   * 
   * @param output
   * @return
   */
  String getLabel(INDArray output);
  
  void getLabel(ArrayList<MLItem> items);

  INDArray output(INDArray features);

  void trainModel() throws Exception;

}
