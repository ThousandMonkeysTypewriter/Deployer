package org.jol.objects.model;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.utilities.DataUtilities;
import org.jol.objects.MLConf;
import org.jol.objects.MLModel;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

public class DL4JModel implements MLModel {

  private DataNormalization normalizer = new NormalizerStandardize();

  private MultiLayerNetwork model;
  private MLConf conf;

  private Map<Integer, String> labels = new HashMap<Integer, String>();

  public DL4JModel (MLConf conf_) throws Exception {
    conf = conf_;
    conf.init();

    if (conf.create) {
      trainModel();
      saveToDisk();
    }
    else 
      restore(conf.modelLocation);

    for ( Entry<Integer, String[]> l : DataUtilities.readEnumCSV(new File(conf.classifier)).entrySet() ) {
      labels.put(l.getKey(), l.getValue()[1]);
    }
  }

  public Model getModel() {
    return model;
  }

  public void trainModel() throws Exception {
    model = (MultiLayerNetwork)conf.train(conf);
  }

  //  public void normalize(INDArray slice) {
  //    normalizer.transform(slice);
  //  }

  public Map<Integer, String> getLabels() {
    return labels;
  }

  public String getLabel(INDArray output) {
    return labels.get(conf.getIndex(output));
  }

  public void restore(String modelLocation) throws IOException {
    model =  ModelSerializer.restoreMultiLayerNetwork(modelLocation);
  }

  public void saveToDisk() throws IOException {
    ModelSerializer.writeModel(model, conf.modelLocation, true);
  }

  public INDArray output(INDArray features) {
    return model.output(features);
  }

  public INDArray prepareFeatures(DataSet test, boolean normalize) throws IOException, InterruptedException {

    if (normalize) {
      fitNormalizer(test);
      normalizer.transform(test);
    }

    return test.getFeatureMatrix();
  }

  public INDArray prepareFeatures(String input) throws IOException, InterruptedException {
    return conf.prepareFeatures(input);
  }

  public void fitNormalizer(DataSet input) {
    normalizer.fit(input);
  }

  //  public Evaluation evaluate(SentimentExampleIterator test) {
  //    return model.evaluate(test);
  //  }

  //  public String eval() throws IOException {
  //    SentimentExampleIterator test = new SentimentExampleIterator(conf.dataPath, wvs, conf.batchSize, conf.truncateReviewsToLength, false);
  //    Evaluation evaluation = model.evaluate(test);
  //    return evaluation.stats();
  //  }
}


