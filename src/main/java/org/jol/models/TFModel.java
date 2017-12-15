package org.jol.models;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.Map.Entry;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.api.Model;
import org.jol.core.MLConf;
import org.jol.core.MLItem;
import org.jol.core.MLModel;
import org.jol.models.dl4j.utilities.DataUtilities;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Tensor;
import org.tensorflow.Session;

public class TFModel implements MLModel {

  private MLConf conf;
  private SavedModelBundle model;
  private Session session;

  public TFModel (MLConf conf_) throws Exception {
    conf = conf_;
    conf.init();

    if (conf.create) {
//      trainModel();
//      saveToDisk();
    }
    else {
      if (conf.modelLocationAbsolute != null &&  new File(conf.modelLocationAbsolute).exists())
        restore(conf.modelLocationAbsolute);
      else 
        throw new RuntimeException("No model was found, please create and save the model first");
    }

//    for ( Entry<Integer, String[]> l : DataUtilities.readEnumCSV(new ClassPathResource(conf.classifier).getFile()).entrySet() ) {
//      labels.put(l.getKey(), l.getValue()[1]);
//    }
  }
  
  @Override
  public void saveToDisk() throws IOException {
    // TODO Auto-generated method stub
    
  }

  @Override
  public void restore(String modelLocation) throws IOException {
    model = SavedModelBundle.load(modelLocation, "serve");
    session = model.session();
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

  public Map<Integer, String> getLabels() {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public Model getModel() {
    return null;
  }

  public String getLabel(INDArray output) {
 /*   Tensor result = session.runner()
        .feed("x_data", inputTensor)
        .fetch("output_layer")
        .run().get(0);

    float[][] m = new float[inputDoubles.size()][1];
    float[][] vector = result.copyTo(m);

    for(int i=0;i<vector.length;i++) 
        items.get(i).score = vector[i][0];		*/
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

  public void prepareFeatures(ArrayList<MLItem> items) {
   for (MLItem i : items) {
     i.setFeatures(null);
   }
  }
  
  public INDArray prepareFeatures(DataSet test, boolean normalize)
      throws IOException, InterruptedException {
    return null;
  }

  @Override
  public void getLabel(ArrayList<MLItem> items) {
    ArrayList<double[]> inputDoubles = new ArrayList<>();

    for ( MLItem item : items ) { 
      double[] dls = new double[conf.numInputs];
      for(int i=0;i<conf.numInputs;i++)  
//        dls[i] = item.getFeatures().get(i);
        dls[i] = 0.0;
      inputDoubles.add(dls);
    }
    
    float[][] matrix = new float[inputDoubles.size()][conf.numInputs];

    for (int j=0;j<inputDoubles.size();j++) 
      for(int i=0;i<conf.batchSizeTest;i++)
        matrix[j][i]=(float)inputDoubles.get(j)[i];
//      items.get(j).tf_matrix = matrix[j];
 
    Tensor inputTensor = Tensor.create(matrix);

    Tensor result = s.runner()
        .feed("x_data", inputTensor)
        .fetch("output_layer")
        .run().get(0);

    float[][] m = new float[inputDoubles.size()][1];
    float[][] vector = result.copyTo(m);

    for(int i=0;i<vector.length;i++)  
//        items.get(i).setLabel(vector[i][0]);
  }

}
