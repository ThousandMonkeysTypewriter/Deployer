package org.jol.models;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Map.Entry;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.api.Model;
import org.jol.core.MLConf;
import org.jol.core.MLModel;
import org.jol.dl4j.utilities.DataUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public class TFModel implements MLModel {

  public TFModel (MLConf conf_) throws Exception {
   /* conf = conf_;
    conf.init();

    if (conf.create) {
      trainModel();
      saveToDisk();
    }
    else {
      if (conf.modelLocationAbsolute != null &&  new File(conf.modelLocationAbsolute).exists())
        restore(conf.modelLocationAbsolute);
      else 
        throw new RuntimeException("No model was found, please create and save the model first");
    }

    for ( Entry<Integer, String[]> l : DataUtilities.readEnumCSV(new ClassPathResource(conf.classifier).getFile()).entrySet() ) {
      labels.put(l.getKey(), l.getValue()[1]);
    }*/
  }
  
  @Override
  public void saveToDisk() throws IOException {
    // TODO Auto-generated method stub
    
  }

  @Override
  public void restore(String modelLocation) throws IOException {
    // TODO Auto-generated method stub
    
  }

  @Override
  public void fitNormalizer(DataSet input) {
    // TODO Auto-generated method stub
    
  }

  @Override
  public INDArray prepareFeatures(String input) throws IOException,
      InterruptedException {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public INDArray prepareFeatures(DataSet test, boolean normalize)
      throws IOException, InterruptedException {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Map<Integer, String> getLabels() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Model getModel() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public String getLabel(INDArray output) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public INDArray output(INDArray features) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public void trainModel() throws Exception {
    // TODO Auto-generated method stub
    
  }

}
